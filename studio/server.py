"""tornado application: login, pages, JSON API, WebSockets.

Every page and API needs a login (pages redirect to ``/login``, the API
answers 401). POSTs need tornado's XSRF token (the pages send it in the
``X-XSRFToken`` header). WebSockets check the login cookie and the origin.

Routes::

    GET  /login  POST /login  POST /logout
    GET  /                         task list, resources, running task, jobs
    GET  /task/<id>                task page (generic, or the task's page.html)
    GET  /api/status               resources, runner, jobs
    GET  /api/tasks                task list
    GET  /api/tasks/<id>           declaration, presets, running state
    POST /api/tasks/<id>/validate  {"values": {...}} -> errors
    POST /api/tasks/<id>/start     {"values": {...}}
    POST /api/tasks/<id>/stop
    GET/POST /api/tasks/<id>/presets          list / save {"name", "values"}
    GET/DELETE /api/tasks/<id>/presets/<name> load / delete
    POST /api/estop                emergency stop of whatever runs
    GET  /api/jobs   POST /api/jobs/<id>/cancel   GET /api/jobs/<id>/log?offset=
    WS   /ws/stream/<id>   WS /ws/control/<id>

Python 3.6 compatible (runs on the Jetson Nano).
"""
import json
import logging
import os

import tornado.template
import tornado.web
from tornado.ioloop import PeriodicCallback

# studio package info
from studio import PACKAGE_DIR, __version__
# Password check, lockout, login cookie
from studio import auth
# Settings and paths
from studio import config
# Parameter validation and presets
from studio import params as params_mod
# Task registry
from studio import tasks as tasks_mod
# Running task, jobs, resources
from studio.jobs import Manager, DeviceBusy
# Runner start failures
from studio.runlink import RunnerError
# WebSockets
from studio.stream import StreamSocket, ControlSocket

# Server log
log = logging.getLogger("studio.server")


# Everything the handlers share
class Studio(object):
    """Settings, tasks, the manager and the login lockout."""

    # Load the tasks and create the manager
    def __init__(self, cfg, manager=None):
        # Settings dict (config.json)
        self.cfg = cfg
        # 'car' or 'pc'
        self.role = config.role(cfg)
        # Settings passed to runners
        self.settings = {"role": self.role, "camera_source": cfg.get("camera_source", "jetbot"),
                         "camera_folder": cfg.get("camera_folder", "")}
        # Task folder overrides from config.json (e.g. lane_data_dir, lane_models_dir, lane_trt_dir)
        self.settings.update((k, v) for k, v in cfg.items() if k.startswith("lane_"))
        # Tasks and broken task packages
        self.tasks, self.task_errors = tasks_mod.discover()
        # id -> task
        self.task_map = dict((t["id"], t) for t in self.tasks)
        # Running task, jobs and resources
        self.manager = manager or Manager(self.settings)
        # Failed-login counter
        self.lockout = auth.Lockout()

    # Whether a task can be started on this machine
    def startable(self, task):
        """Return ``(ok, reason)``."""
        # Still in development
        if task["status"] != "ready":
            # Refuse
            return False, "this task is still in development"
        # Not meant for this machine
        if self.role not in task["roles"]:
            # Refuse
            return False, "this task runs on: %s (this machine is '%s')" % (", ".join(task["roles"]), self.role)
        # Allowed
        return True, ""


# Base of every page / API handler
class BaseHandler(auth.CurrentUserMixin, tornado.web.RequestHandler):
    """Shared helpers."""

    # Store the studio object
    def initialize(self, studio):
        # Shared state
        self.studio = studio

    # Write a JSON response
    def write_json(self, obj, status=200):
        """Send ``obj`` as JSON with ``status``."""
        # HTTP status
        self.set_status(status)
        # Content type
        self.set_header("Content-Type", "application/json; charset=utf-8")
        # Never cached
        self.set_header("Cache-Control", "no-store")
        # ASCII JSON
        self.finish(json.dumps(obj))

    # Request body as JSON
    def json_body(self):
        """Parsed body (empty dict when there is none); 400 on bad JSON."""
        # No body
        if not self.request.body:
            # Empty
            return {}
        # Parse it
        try:
            # JSON object
            data = json.loads(self.request.body.decode("utf-8"))
        # Not JSON
        except ValueError:
            # Bad request
            raise tornado.web.HTTPError(400, "body is not JSON")
        # Must be an object
        if not isinstance(data, dict):
            # Bad request
            raise tornado.web.HTTPError(400, "body must be a JSON object")
        # The object
        return data

    # Task by id or 404
    def get_task(self, task_id):
        """The normalised task or HTTP 404."""
        # Look it up
        task = self.studio.task_map.get(task_id)
        # Unknown
        if task is None:
            # Not found
            raise tornado.web.HTTPError(404, "unknown task")
        # Found
        return task

    # Errors of API requests as JSON
    def write_error(self, status_code, **kwargs):
        # API: JSON body
        if self.request.path.startswith("/api/"):
            # Reason text
            reason = self._reason
            # HTTPError with a log message: use it as the reason
            exc = kwargs.get("exc_info", (None, None, None))[1]
            # Our own HTTPErrors carry a readable message
            if isinstance(exc, tornado.web.HTTPError) and exc.log_message:
                # Readable message
                reason = exc.log_message
            # JSON error
            self.finish(json.dumps({"error": reason}))
        # Pages: tornado's default page
        else:
            # Default
            tornado.web.RequestHandler.write_error(self, status_code, **kwargs)


# Decorator for API methods: 401 JSON instead of a redirect
def api_auth(method):
    """Require a login; answer 401 JSON otherwise."""
    # Wrapped method
    def wrapper(self, *args, **kwargs):
        # Not logged in
        if not self.current_user:
            # Unauthorized
            raise tornado.web.HTTPError(401, "login required")
        # Run the method
        return method(self, *args, **kwargs)
    # Keep the name for debugging
    wrapper.__name__ = method.__name__
    # The wrapper
    return wrapper


# Login page and form
class LoginHandler(BaseHandler):
    # Show the form
    def get(self):
        # Already logged in: home
        if self.current_user:
            # Redirect
            self.redirect(self.get_argument("next", "/"))
            # Done
            return
        # Form without an error
        self.render("login.html", error=None, next_url=self.get_argument("next", "/"))

    # Check the password
    def post(self):
        # Client address (lockout key)
        addr = self.request.remote_ip
        # Where to go afterwards (only local paths)
        next_url = self.get_argument("next", "/")
        # Never redirect to another site
        if not next_url.startswith("/") or next_url.startswith("//"):
            # Home instead
            next_url = "/"
        # Locked address
        wait = self.studio.lockout.check(addr)
        # Refuse while locked
        if wait > 0:
            # Too many requests
            self.set_status(429)
            # Form with the waiting time
            self.render("login.html", error="locked", wait=int(wait + 1), next_url=next_url)
            # Done
            return
        # Check the password
        if auth.verify_password(self.get_argument("password", ""), self.studio.cfg.get("password")):
            # Clear the failures
            self.studio.lockout.success(addr)
            # Signed cookie (single shared password: the user name is fixed)
            self.set_login("operator")
            # Server log
            log.info("login from %s", addr)
            # Go on
            self.redirect(next_url)
            # Done
            return
        # Count the failure
        self.studio.lockout.fail(addr)
        # Server log
        log.warning("failed login from %s", addr)
        # Forbidden
        self.set_status(403)
        # Form with the error
        self.render("login.html", error="wrong", wait=0, next_url=next_url)


# Logout
class LogoutHandler(BaseHandler):
    # Remove the cookie
    def post(self):
        # Delete the login cookie
        self.clear_cookie(auth.COOKIE_NAME)
        # Back to the login page
        self.redirect("/login")


# Home page
class IndexHandler(BaseHandler):
    # Task list
    @tornado.web.authenticated
    def get(self):
        # Page with the tasks (status is loaded by JavaScript)
        self.render("index.html", tasks=self.studio.tasks, errors=self.studio.task_errors,
                    startable=self.studio.startable)


# Task page
class TaskPageHandler(BaseHandler):
    # Generic page or the task's own template
    @tornado.web.authenticated
    def get(self, task_id):
        # The task
        task = self.get_task(task_id)
        # Whether it may start here
        ok, reason = self.studio.startable(task)
        # Task-specific body, if any
        body = None
        # Render the task's page.html into the generic frame
        if task["page"]:
            # Load it with the same loader settings
            loader = tornado.template.Loader(os.path.dirname(task["page"]))
            # Rendered HTML
            body = loader.load(os.path.basename(task["page"])).generate(
                task=task, handler=self, xsrf_form_html=self.xsrf_form_html, static_url=self.static_url)
        # The page
        self.render("task.html", task=task, task_json=json.dumps(task).replace("</", "<\\/"), startable=ok, reason=reason,
                    body=body)


# Status API
class StatusApi(BaseHandler):
    # Resources, runner, jobs
    @api_auth
    def get(self):
        # Manager status plus the version
        status = self.studio.manager.status()
        # Version and role
        status.update({"version": __version__, "role": self.studio.role})
        # Send it
        self.write_json(status)


# Task list API
class TasksApi(BaseHandler):
    # Every task
    @api_auth
    def get(self):
        # Declarations with the startable flag
        out = []
        # Every task
        for t in self.studio.tasks:
            # Whether it can start here
            ok, reason = self.studio.startable(t)
            # Short form
            out.append({"id": t["id"], "title": t["title"], "description": t["description"],
                        "status": t["status"], "startable": ok, "reason": reason})
        # Send them
        self.write_json({"tasks": out, "errors": self.studio.task_errors})


# One task API
class TaskApi(BaseHandler):
    # Declaration, presets and running state
    @api_auth
    def get(self, task_id):
        # The task
        task = self.get_task(task_id)
        # Whether it can start here
        ok, reason = self.studio.startable(task)
        # Running link, if any
        link = self.studio.manager.link_for(task_id)
        # Send it
        self.write_json({"task": task, "presets": params_mod.list_presets(task_id), "startable": ok,
                         "reason": reason, "runner": link.describe() if link else None})


# Validate parameters
class ValidateApi(BaseHandler):
    # Check values without starting
    @api_auth
    def post(self, task_id):
        # The task
        task = self.get_task(task_id)
        # Validate
        clean, errors = params_mod.validate(task["params"], self.json_body().get("values"))
        # Result
        self.write_json({"values": clean, "errors": errors})


# Start a task
class StartApi(BaseHandler):
    # Validate, then start the runner
    @api_auth
    async def post(self, task_id):
        # The task
        task = self.get_task(task_id)
        # Allowed on this machine?
        ok, reason = self.studio.startable(task)
        # Refuse
        if not ok:
            # Conflict with the reason
            self.write_json({"error": reason}, 409)
            # Done
            return
        # Validate the values
        clean, errors = params_mod.validate(task["params"], self.json_body().get("values"))
        # Invalid values: never start
        if errors:
            # Bad request with the field errors
            self.write_json({"error": "invalid parameters", "errors": errors}, 400)
            # Done
            return
        # Start it
        try:
            # Runner link
            link = await self.studio.manager.start_task(task, clean, user=self.current_user)
        # Another task holds the devices
        except DeviceBusy as e:
            # Conflict
            self.write_json({"error": str(e)}, 409)
            # Done
            return
        # Runner failed to start (camera busy, import error, ...)
        except RunnerError as e:
            # Server error with the runner's message
            self.write_json({"error": str(e)}, 500)
            # Done
            return
        # Running
        self.write_json({"runner": link.describe()})


# Stop a task
class StopApi(BaseHandler):
    # Stop if it is the running one
    @api_auth
    async def post(self, task_id):
        # The task must exist
        self.get_task(task_id)
        # Running link of this task
        if self.studio.manager.link_for(task_id) is None:
            # Not running
            self.write_json({"stopped": False})
            # Done
            return
        # Stop it
        await self.studio.manager.stop_task()
        # Stopped
        self.write_json({"stopped": True})


# Preset list / save
class PresetsApi(BaseHandler):
    # Names
    @api_auth
    def get(self, task_id):
        # The task must exist
        self.get_task(task_id)
        # Names
        self.write_json({"presets": params_mod.list_presets(task_id)})

    # Save
    @api_auth
    def post(self, task_id):
        # The task
        task = self.get_task(task_id)
        # Name and values
        body = self.json_body()
        # Validate and store
        ok, errors = params_mod.save_preset(task_id, task["params"], body.get("name"), body.get("values"))
        # Result
        self.write_json({"saved": ok, "errors": errors, "presets": params_mod.list_presets(task_id)},
                        200 if ok else 400)


# One preset
class PresetApi(BaseHandler):
    # Load
    @api_auth
    def get(self, task_id, name):
        # The task
        task = self.get_task(task_id)
        # Values (re-validated)
        values, errors = params_mod.load_preset(task_id, task["params"], name)
        # Missing preset
        if values is None:
            # Not found
            self.write_json({"error": errors.get("name")}, 404)
            # Done
            return
        # Values and dropped / invalid fields
        self.write_json({"values": values, "errors": errors})

    # Delete
    @api_auth
    def delete(self, task_id, name):
        # The task must exist
        self.get_task(task_id)
        # Remove it
        self.write_json({"deleted": params_mod.delete_preset(task_id, name),
                         "presets": params_mod.list_presets(task_id)})


# Emergency stop of whatever task runs
class EstopApi(BaseHandler):
    # Stop now
    @api_auth
    def post(self):
        # Running link of any task
        r = self.studio.manager.runner
        # Forward when it runs
        sent = r is not None and r.estop()
        # Result
        self.write_json({"sent": bool(sent)})


# Job list
class JobsApi(BaseHandler):
    # Newest first
    @api_auth
    def get(self):
        # Jobs
        self.write_json({"jobs": self.studio.manager.jobs.describe()})


# Cancel a job
class JobCancelApi(BaseHandler):
    # Cancel
    @api_auth
    def post(self, job_id):
        # Result
        self.write_json({"cancelled": self.studio.manager.jobs.cancel(job_id)})


# Part of a job's log
class JobLogApi(BaseHandler):
    # From an offset
    @api_auth
    def get(self, job_id):
        # Requested offset
        offset = int(self.get_argument("offset", "0") or 0)
        # Text and next offset
        text, nxt = self.studio.manager.jobs.read_log(job_id, offset)
        # Unknown job
        if text is None:
            # Not found
            self.write_json({"error": "unknown job"}, 404)
            # Done
            return
        # Result
        self.write_json({"text": text, "offset": nxt})


# Build the tornado application
def make_app(studio, debug=False):
    """``tornado.web.Application`` for a ``Studio``."""
    # Arguments every handler gets
    a = {"studio": studio}
    # Core routes
    routes = [
        (r"/login", LoginHandler, a), (r"/logout", LogoutHandler, a), (r"/", IndexHandler, a),
        (r"/task/([a-z0-9_]+)", TaskPageHandler, a),
        (r"/api/status", StatusApi, a), (r"/api/tasks", TasksApi, a),
        (r"/api/tasks/([a-z0-9_]+)", TaskApi, a),
        (r"/api/tasks/([a-z0-9_]+)/validate", ValidateApi, a),
        (r"/api/tasks/([a-z0-9_]+)/start", StartApi, a),
        (r"/api/tasks/([a-z0-9_]+)/stop", StopApi, a),
        (r"/api/tasks/([a-z0-9_]+)/presets", PresetsApi, a),
        (r"/api/tasks/([a-z0-9_]+)/presets/([A-Za-z0-9_\-]+)", PresetApi, a),
        (r"/api/estop", EstopApi, a), (r"/api/jobs", JobsApi, a),
        (r"/api/jobs/([A-Za-z0-9_\-x]+)/cancel", JobCancelApi, a),
        (r"/api/jobs/([A-Za-z0-9_\-x]+)/log", JobLogApi, a),
        (r"/ws/stream/([a-z0-9_]+)", StreamSocket, a), (r"/ws/control/([a-z0-9_]+)", ControlSocket, a),
    ]
    # Task-specific handlers (their own pages / APIs)
    for t in studio.tasks:
        # A broken web.py is reported, not fatal
        try:
            # The task's web module, if any
            mod = tasks_mod.web_module(t["id"])
            # No extra pages
            if mod is None:
                # Next task
                continue
            # Add each handler with the same arguments
            for pattern, handler in getattr(mod, "HANDLERS", []):
                # Route
                routes.append((pattern, handler, a))
            # One-time setup (job parsers, hooks)
            if hasattr(mod, "setup"):
                # Give it the shared state
                mod.setup(studio)
        # Record it with the task errors
        except Exception as e:
            # Shown on the home page
            studio.task_errors[t["id"] + ".web"] = repr(e)
    # The application
    return tornado.web.Application(
        routes,
        # Page templates
        template_path=os.path.join(PACKAGE_DIR, "templates"),
        # CSS / JavaScript
        static_path=os.path.join(PACKAGE_DIR, "static"),
        # Signed login cookie
        cookie_secret=studio.cfg["cookie_secret"],
        # Days a login stays valid
        login_days=studio.cfg.get("login_days", 7),
        # Where @authenticated redirects
        login_url="/login",
        # Cross-site request forgery protection for POST / DELETE
        xsrf_cookies=True,
        # Reload templates while developing
        debug=debug,
        # Version shown in the footer
        studio_version=__version__,
    )


# Start the periodic jobs of a studio (job queue)
def start_timers(studio):
    """Advance the job queue once per second; returns the timer."""
    # Periodic queue tick
    timer = PeriodicCallback(studio.manager.jobs.tick, 1000)
    # Start it
    timer.start()
    # The timer
    return timer

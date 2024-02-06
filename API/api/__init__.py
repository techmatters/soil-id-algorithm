import os

from flask import Flask


def create_app(config, debug=False, testing=False, config_overrides=None):
    app = Flask(__name__)
    environment = os.environ.get("FLASK_CONFIG", "config")

    app.config.from_object("config.{}".format(environment.capitalize()))

    from v3.routes import API_VERSION_V3, api_v3_bp

    app.register_blueprint(
        api_v3_bp,
        url_prefix="{prefix}/v{version}".format(
            prefix=app.config["URL_PREFIX"], version=API_VERSION_V3
        ),
    )
    return app

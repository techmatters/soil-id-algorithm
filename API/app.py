#!/usr/bin/env python2.7
import api
import config
from flask_cors import CORS

app = api.create_app(config)
CORS(app)

if __name__ == "__main__":
    # app.run()
    app.run(host=app.config["HOST"], port=app.config["PORT"], debug=True)

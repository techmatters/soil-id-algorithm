import api
import config
from flask_cors import CORS
from twisted.internet import reactor
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.wsgi import WSGIResource

app = api.create_app(config)
CORS(app)


# @app.route('/example')
def index():
    return "LandPKS SoilID API"


flask_site = WSGIResource(reactor, reactor.getThreadPool(), app)

root = Resource()
root.putChild("api", flask_site)

reactor.listenTCP(app.config["PORT"], Site(root))
reactor.run()

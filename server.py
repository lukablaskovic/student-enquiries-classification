from text_categorizer import topicClassifier
from aiohttp import web
import asyncio
import aiohttp_cors

routes = web.RouteTableDef()


@routes.get("/")
async def get_handler(request):
    return web.json_response({"status": "OK"}, status=200)


@routes.post("/classificator")
async def classify(request):
    try:
        inquiry = await request.json()
        response = topicClassifier(inquiry["text"])
        print(response)
        return web.json_response(response, status=200)
    except Exception as e:
        return web.json_response({"status": "failed", "message": str(e)}, status=500)

app = web.Application()
app.router.add_routes(routes)

# Cors policy set up
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*"
    )
})
for route in list(app.router.routes()):
    cors.add(route)


if __name__ == '__main__':
    web.run_app(app)

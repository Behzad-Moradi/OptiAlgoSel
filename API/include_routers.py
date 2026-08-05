
from API.routers import prediction, welcome, algorithms, health, model, result


def include_routers(app):
    for router in (
        welcome.router,
        health.router,
        algorithms.router,
        model.router,
        prediction.router,
        result.router    
        ):
        app.include_router(router)
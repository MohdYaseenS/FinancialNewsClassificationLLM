from backend.api import app
import uvicorn
from configs.config import BACKEND_HOST, BACKEND_PORT

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=False
    )
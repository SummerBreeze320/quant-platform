import uvicorn
from src.common.config import get_settings
from src.common.logger import logger

settings = get_settings()

def main():
    """Main application entrypoint launching FastAPI server with Uvicorn."""
    logger.info(f"Starting {settings.PROJECT_NAME} in '{settings.ENV}' mode...")
    uvicorn.run(
        "src.service.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()

import os

class Config:
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-dev-key')
    DATABASE_URI = os.getenv('DB_URI', 'postgresql://user:pass@localhost/dbname')
    API_KEY = os.getenv('API_KEY', 'your-api-key-here')
    
    @staticmethod
    def validate():
        """检查生产环境关键配置"""
        if os.getenv('FLASK_ENV') == 'production':
            assert not Config.DEBUG, "DEBUG must be False in production"
            assert Config.SECRET_KEY != 'default-dev-key', "Set unique SECRET_KEY"
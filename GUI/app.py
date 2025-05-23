from flask import Flask
from blueprints.index import bp as index_bp
from blueprints.functionsPage import functions_bp


def create_app():
    app = Flask(__name__)
    app.secret_key = 'gr123456'  # 设置 session 密钥

    # 注册蓝图
    app.register_blueprint(index_bp)
    app.register_blueprint(functions_bp)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
from ml import create_app

if __name__ == "__main__":
    # app = create_app('development')
    app = create_app()
    # print(app.root_path)
    # print(app.instance_path)
    app.run(port=app.config['PORT'])

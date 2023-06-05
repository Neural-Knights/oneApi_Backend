from flask import Flask, Blueprint, jsonify, request
from recommendation import recommend_by_course_title

recomend = Blueprint('recomend', __name__)

@recomend.route('/recommend', methods=['POST'])
def recommend_course():
    title = request.json['title']
    recommended_courses = recommend_by_course_title(title)
    return jsonify(recommended_courses.to_dict('records'))

def create_app():
    app = Flask(__name__)
    app.register_blueprint(recomend)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run()

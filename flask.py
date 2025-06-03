from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import re
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import subprocess

app = Flask(__name__)

# 配置
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'garment_embeds'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """首页 - 返回HTML页面"""
    return render_template('index.html')

@app.route('/api/garments', methods=['GET'])
def get_garments():
    """GET接口 - 获取列表"""
    def get_unique_prefixes(directory):
        prefixes = set()
        pattern = re.compile(r"^(\d+)_\d+\.safetensors$")
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                prefixes.add(match.group(1))
        return sorted(list(prefixes))
    garment_sft_paths = get_unique_prefixes("garment_embeds")
    arr = []
    for sft in garment_sft_paths:
        arr.append({"sft": sft, "image": sft + ".jpg"})
    return jsonify({
        "garments": arr,
    })

@app.route('/api/users', methods=['POST'])
def create_user():
    """POST接口 - 创建新用户"""
    data = request.get_json()
    
    if not data or not data.get('name') or not data.get('email'):
        return jsonify({
            "status": "error",
            "message": "姓名和邮箱为必填项"
        }), 400
    
    new_user = {
        "id": 1,
        "name": data['name'],
        "email": data['email']
    }
    
    return jsonify({
        "status": "success",
        "message": "用户创建成功",
        "data": new_user
    }), 201

@app.route('/api/garment', methods=['POST'])
def upload_file():
    """POST接口 - 上传服装图片"""
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No garment image"
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "没有选择文件"
        }), 400
    img_path = app.config['UPLOAD_FOLDER'] + "/" + file.filename
    embeds_path = app.config['UPLOAD_FOLDER'] + "/" + file.filename.replace(".jpg", ".safetensors")
    file.save(img_path)
    try:
        result = subprocess.run(
            ['python process_garment_embeds.py', '--image_path ' + img_path, "--save_path " + embeds_path],     # 替换成你的命令
            capture_output=True,
            text=True,
            check=True                        # 自动抛出异常 on failure
        )
        print("命令成功执行")
        print("输出：", result.stdout)
        return jsonify({
            "status": "OK",
            "message": ""
        }), 200
    except subprocess.CalledProcessError as e:
        print("命令执行失败")
        print("返回码：", e.returncode)
        print("错误输出：", e.stderr)
        return jsonify({
            "status": "error",
            "message": "Error"
        }), 400
    except Exception as e:
        print("其他错误：", str(e))
        return jsonify({
            "status": "error",
            "message": "Error"
        }), 400

@app.route('/garments/<filename>')
def uploaded_file(filename):
    """静态文件服务 - 访问上传的图片"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        "status": "error",
        "message": "Unknown path"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        "status": "error",
        "message": "Error"
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
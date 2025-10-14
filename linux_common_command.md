# 修改默认pip为清华镜像
```bash
##永久
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 验证配置
pip config list

##一次性 
pip install <库名> -i https://pypi.tuna.tsinghua.edu.cn/simple #清华url
```

# 设置环境变量
```bash
# 编辑配置文件
nano ~/.bashrc

# 在文件末尾添加：
export DEEPSEEK_API_KEY="sk-37ffd9c97b9444888aa2ad45b88c23a5"

# 保存后重新加载
source ~/.bashrc

#验证环境变量是否设置成功
echo $DEEPSEEK_API_KEY

#保存并退出：
按 Ctrl + O （字母O，不是数字0）
按 Enter 确认文件名
按 Ctrl + X 退出编辑器

```
# 导入环境到requirements.txt
```bash
pip freeze > requirement.txt #导入环境
```
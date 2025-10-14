# Git 常用指令速查手册

## 配置相关

### 配置用户信息
```bash
git config --global user.name "你的用户名"  
git config --global user.email "你的邮箱"
```

### 查看配置
```bash
git config --list
```

## 仓库初始化

### 初始化本地仓库
```bash
git init
```
## 不需要 git init 的情况：
```bash
从远程仓库克隆（git clone 会自动初始化）

已有 .git 文件夹的目录

子目录（Git 会自动识别父目录的仓库）
```
### 克隆远程仓库
```bash
git clone <仓库地址>
```

## 查看状态

### 查看工作区状态
```bash
git status
```

### 查看提交历史
```bash
git log                    # 详细格式
git log --oneline         # 简洁格式
git log --oneline -5      # 最近5次提交
```

## 文件操作

### 添加文件（工作区 → 暂存区）
```bash
git add <文件名>          # 添加单个文件
git add .                 # 添加所有修改的文件
git add -A                # 添加所有文件（包括新文件）
```

### 提交文件（暂存区 → 本地仓库）
```bash
git commit -m "提交信息"    # 提交并添加描述 提交信息可以在log查到
git commit --amend        # 修改最后一次提交
git commit --amend -m "新的提交信息" # 或者直接修改提交信息而不打开编辑器
```

## 分支操作
```bash
git branch                    # 查看所有分支
git branch <分支名>           # 创建新分支
git checkout <分支名>         # 切换分支
git checkout -b <分支名>      # 创建并切换分支
git branch -M <新分支名>      # 重命名当前分支
```

## 远程仓库操作
```bash
git remote add origin <远程地址>        # 添加远程仓库 origin是远程仓库的别名
git remote -v                         # 查看远程仓库
git remote set-url origin <新地址>      # 设置远程仓库地址
```

## 同步操作
```bash
git pull origin <分支名>        # 从远程拉取
git push origin <分支名>        # 推送到远程
git push -u origin <分支名>     # 首次推送并设置上游分支
git push                        # 推送到上游分支（设置-u后）
```

## 查看差异
```bash
git diff                        # 查看工作区与暂存区的差异
git diff --cached              # 查看暂存区与本地仓库的差异
```

## 撤销操作
```bash
git checkout -- <文件名>        # 撤销工作区的修改
git reset HEAD <文件名>         # 撤销暂存区的文件（回到工作区）
git reset --hard HEAD~1         # 撤销最后一次提交
```

## 日常工作流程
```bash
# 1. 开始工作前同步远程
git pull

# 2. 修改文件后添加
git add .

# 3. 提交修改
git commit -m "修改描述"

# 4. 推送到远程
git push
```

# pip指令
```bash
# -index-url https://download.pytorch.org/whl/cpu
```
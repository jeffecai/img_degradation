#!/bin/bash
# 设置GitHub远程仓库脚本
# 使用方法：./setup_remote.sh YOUR_GITHUB_USERNAME

if [ -z "$1" ]; then
    echo "错误：请提供您的GitHub用户名"
    echo "使用方法: ./setup_remote.sh YOUR_GITHUB_USERNAME"
    exit 1
fi

GITHUB_USERNAME=$1
REPO_NAME="img_degrade"

echo "正在设置远程仓库..."
git remote add origin "git@github.com:${GITHUB_USERNAME}/${REPO_NAME}.git"

echo "远程仓库已设置："
git remote -v

echo ""
echo "现在可以执行以下命令推送代码："
echo "  git push -u origin main"
echo ""
echo "或者如果您的默认分支是master："
echo "  git push -u origin master"


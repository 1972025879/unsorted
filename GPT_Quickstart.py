import os
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v3.1_terminus_expires_on_20251015"
)

def chat_with_reasoning(messages, user_input=None):
    """
    与DeepSeek Reasoning模型对话
    
    Args:
        messages: 对话历史列表
        user_input: 用户输入内容，如果为None则从终端读取
    
    Returns:
        tuple: (更新后的messages, 推理过程, 生成内容)
    """
    if user_input is None:
        user_input = input("\n请输入你的问题: ")
    
    # 添加用户消息到对话历史
    messages.append({"role": "user", "content": user_input})
    
    # 调用API
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
    )
    
    # 获取推理过程和生成内容
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    
    # 添加助理回复到对话历史
    messages.append({"role": "assistant", "content": content})
    
    return messages, reasoning_content, content

def main():
    """主函数，实现多轮对话"""
    print("=== DeepSeek Reasoning 对话系统 ===")
    print("输入 'quit' 或 '退出' 结束对话\n")
    
    # 初始化对话历史
    messages = []
    
    while True:
        try:
            # 进行对话
            messages, reasoning, answer = chat_with_reasoning(messages)
            
            # 显示结果
            print("\n" + "="*50)
            print("🤔 推理过程:")
            print(reasoning)
            print("\n💡 最终答案:")
            print(answer)
            print("="*50)
            
            # 检查是否退出
            user_input = messages[-2]["content"]  # 获取最后一次用户输入
            if user_input.lower() in ['quit', '退出', 'exit']:
                print("对话结束！")
                break
                
        except KeyboardInterrupt:
            print("\n\n对话被用户中断！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            break

# 更简洁的版本（不显示推理过程）
def simple_chat():
    """简化版对话，只显示最终答案"""
    print("=== DeepSeek Reasoning 简易对话 ===")
    print("输入 'quit' 或 '退出' 结束对话\n")
    
    messages = []
    
    while True:
        try:
            user_input = input("\n用户: ")
            
            if user_input.lower() in ['quit', '退出', 'exit']:
                print("对话结束！")
                break
            
            messages, reasoning, answer = chat_with_reasoning(messages, user_input)
            print(f"\n助手: {answer}")
            
        except KeyboardInterrupt:
            print("\n\n对话结束！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            break

if __name__ == "__main__":
    # 选择对话模式
    print("请选择对话模式:")
    print("1. 完整模式（显示推理过程）")
    print("2. 简易模式（只显示答案）")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        main()
    else:
        simple_chat()
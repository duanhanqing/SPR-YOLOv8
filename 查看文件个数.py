import os

def count_files(directory):
    file_count = 0
    
    # 遍历目录中的所有项目
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # 如果是文件，计数器加1
        if os.path.isfile(item_path):
            file_count += 1
    
    return file_count

if __name__ == "__main__":
    # 设置要统计的文件夹路径
    folder_path = "/szzn/luoluo/work2/YOLOv8.2/datasets/BDD_100K/images/train"  # 替换为你的文件夹路径
    
    # 验证文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
    else:
        total_files = count_files(folder_path)
        print(f"Total files in '{folder_path}': {total_files}")
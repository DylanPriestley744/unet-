import matplotlib.pyplot as plt

def plot_picture(filename):
    try:
        with open(filename, 'r') as f:
            train_loss = f.readlines()
            train_loss = list(map(lambda x: float(x.strip()), train_loss))
        x = range(len(train_loss))
        y = train_loss
        plt.plot(x, y, label='train loss', linewidth=2, color='r', marker='o', markerfacecolor='r', markersize=5)
        plt.xlabel('epoch')
        plt.ylabel('loss value')
        plt.legend()
        # 显示图像
        plt.show()
        # 保存图像到文件
        plt.savefig('train_loss_plot.png')
        print("损失曲线已保存为 train_loss_plot.png")
    except FileNotFoundError:
        print(f"文件 {filename} 不存在，请检查路径是否正确。")
    except ValueError:
        print(f"文件 {filename} 的内容格式有误，请检查是否为每行一个损失值。")

if __name__ == "__main__":
    # 修改为绝对路径
    plot_picture(r'd:\unet\Unet_train.txt')
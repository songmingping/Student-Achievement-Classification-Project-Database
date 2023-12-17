import socket


def connect_to_server(server_ip, server_port):
    try:
        # 创建客户端套接字
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 连接到服务器
        client_socket.connect((server_ip, server_port))

        print("成功连接到服务器")

        # 在此处可以进行与服务器的通信操作，发送或接收数据

        # 关闭客户端套接字
        client_socket.close()

    except socket.error as e:
        print("连接错误:", e)


# 在这里替换为您的服务器 IP 地址和端口号
server_ip = '172.20.10.2'
server_port = 9002

# 调用连接函数
connect_to_server(server_ip, server_port)

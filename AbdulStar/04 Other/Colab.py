# from google.colab import drive
# drive.mount('/content/gdrive')

# !cp "/content/datafull.pkl" "/content/gdrive/MyDrive/Data"

# !cat /proc/cpuinfo

# from psutil import virtual_memory
# ram_gb = virtual_memory().total / 1e9
# print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

# gpu_info = !nvidia-smi
# gpu_info = '\n'.join(gpu_info)
# if gpu_info.find('failed') >= 0:
#   print('Not connected to a GPU')
# else:
#   print(gpu_info)
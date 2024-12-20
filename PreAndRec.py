import matplotlib.pyplot as plt
# pointnet:
xy0 = [[0, 1], [0.1, 0.927], [0.2, 0.920], [0.3, 0.916], [0.4, 0.911], [0.5, 0.899], [0.6, 0.865], [0.7, 0.843], [0.8, 0.787], [0.9, 0.609], [1.0, 0.091]]
# pointnet++:
xy1 = [[0, 1], [0.1, 0.969], [0.2, 0.967], [0.3, 0.966], [0.4, 0.965], [0.5, 0.953], [0.6, 0.921], [0.7, 0.895], [0.8, 0.839], [0.9, 0.693], [1.0, 0.058]]
# dgcnn:
xy2 = [[0, 1], [0.1, 0.996], [0.2, 0.994], [0.3, 0.994], [0.4, 0.993], [0.5, 0.987], [0.6, 0.972], [0.7, 0.958], [0.8, 0.899], [0.9, 0.830], [1.0, 0.046]]
# PCT:
xy3 = [[0, 1], [0.1, 0.937], [0.2, 0.932], [0.3, 0.930], [0.4, 0.924], [0.5, 0.915], [0.6, 0.893], [0.7, 0.860], [0.8, 0.786], [0.9, 0.647], [1.0, 0.113]]
# CF3d:
xy4 = [[0, 1], [0.1, 0.997], [0.2, 0.996], [0.3, 0.996], [0.4, 0.995], [0.5, 0.990], [0.6, 0.989], [0.7, 0.974], [0.8, 0.962], [0.9, 0.955], [1.0, 0.142]]
list1 = [xy0, xy1, xy2, xy3, xy4]
n = 0
# # scanobject
# # pointnet:     xy = [[0, 1], [0.1, 0.927], [0.2, 0.920], [0.3, 0.916], [0.4, 0.911], [0.5, 0.899], [0.6, 0.865], [0.7, 0.843], [0.8, 0.787], [0.9, 0.609], [1.0, 0.091]]
# # pointnet++:   xy = [[0, 1], [0.1, 0.969], [0.2, 0.967], [0.3, 0.966], [0.4, 0.965], [0.5, 0.953], [0.6, 0.921], [0.7, 0.895], [0.8, 0.839], [0.9, 0.693], [1.0, 0.058]]
# # dgcnn:        xy = [[0, 1], [0.1, 0.996], [0.2, 0.994], [0.3, 0.994], [0.4, 0.993], [0.5, 0.987], [0.6, 0.972], [0.7, 0.958], [0.8, 0.899], [0.9, 0.830], [1.0, 0.046]]
# # PCT:          xy = [[0, 1], [0.1, 0.937], [0.2, 0.932], [0.3, 0.930], [0.4, 0.924], [0.5, 0.915], [0.6, 0.893], [0.7, 0.860], [0.8, 0.786], [0.9, 0.647], [1.0, 0.113]]
# # CF3d:         xy = [[0, 1], [0.1, 0.997], [0.2, 0.996], [0.3, 0.996], [0.4, 0.995], [0.5, 0.990], [0.6, 0.989], [0.7, 0.974], [0.8, 0.962], [0.9, 0.955], [1.0, 0.142]]
for num in list1:
    x = []
    y = []
    for i, m in enumerate(num):
    #   print(i,m)
      x.append(m[0])
      y.append(m[1])
    print(x)
    print(y)
    if n == 0:
        plt.plot(x, y, color='black', label='PointNet', linewidth=1.0, marker = '$\\bigodot$' )
    elif n == 1:
        plt.plot(x, y, color='green', label='PointNet++', linewidth=1.0, marker='s')
    elif n == 2:
        plt.plot(x, y, color='red', label='DGCNN', linewidth=1.0, marker='*')
    elif n == 3:
        plt.plot(x, y, color='pink', label='PCT', linewidth=1.0, marker='x')
    else:
        plt.plot(x, y, color='blue', label='CF3d', linewidth=1.0, marker='$\Delta$')
    n = n + 1

plt.legend(['PointNet','PointNet++','DGCNN','PCT','CF3d'])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
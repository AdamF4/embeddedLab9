import os
import time

prev_time = 0
last_image = 0
while True:
    current_time = time.time()
    if prev_time < current_time - 2:
        new_images = []
        prev_time = current_time
        for file in os.listdir("task7_images"):
            if file.endswith(".jpg"):
                i = int(file.replace('.jpg', ''))
                if i > last_image:
                    # new_images.append(os.path.join("task7_images/", file))
                    # print(os.path.join("task7_images/", file))
                    last_image = max(i, last_image)

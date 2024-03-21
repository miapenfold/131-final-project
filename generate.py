import numpy as np
import constants
import cv2

class Generate:
    def __init__(self, height, width, building_image_path):
        # pipe: [[(x: position, y: top, y: bottom)]]
        self.pipes = []
        self.height = height
        self.width = width
        self.points = 0
        self.building_image = cv2.imread(building_image_path, cv2.IMREAD_UNCHANGED)

    def create(self):
        rand_y_top = np.random.randint(0, self.height - constants.GAP)
        self.pipes.append(
            [self.width, rand_y_top, rand_y_top + constants.GAP, False]  # The last 'False' is the new flag indicating the pipe hasn't been passed
         )

        
    def place_image_on_frame(self, frame, image, top_left_x, top_left_y, target_width, target_height, pipe_index):
        # Ensure there is any space to place the image
        if target_width <= 0 or target_height <= 0 or top_left_x >= frame.shape[1] or top_left_y >= frame.shape[0]:
            return  # Nothing to place because the image or the placement is out of bounds

        # Resize the image to the target size
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Check if the resized image has dimensions
        if resized_image.size == 0:
            return  # The resized image has zero size, so nothing to overlay

        # If the image has an alpha channel, use it for blending
        if resized_image.shape[2] == 4:
            alpha_s = resized_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                target_slice = frame[top_left_y:top_left_y + target_height, top_left_x:top_left_x + target_width, c]
                # Check if target slice dimensions match resized image dimensions before operation
                if target_slice.shape[0] == resized_image.shape[0] and target_slice.shape[1] == resized_image.shape[1]:
                    frame[top_left_y:top_left_y + target_height, top_left_x:top_left_x + target_width, c] = \
                        alpha_s * resized_image[:, :, c] + alpha_l * target_slice
                else:
                    return  # Avoid the operation as dimensions don't match
        else:
            # Place the resized image onto the frame, ensuring dimensions match
            target_slice = frame[top_left_y:top_left_y + target_height, top_left_x:top_left_x + target_width]
            if target_slice.shape[0] == resized_image.shape[0] and target_slice.shape[1] == resized_image.shape[1]:
                frame[top_left_y:top_left_y + target_height, top_left_x:top_left_x + target_width] = resized_image


            
    def draw_pipes(self, frm):
        for index, i in enumerate(self.pipes):
            pipe_right_x = i[0] + constants.PIPE_WIDTH
            if pipe_right_x > frm.shape[1]:
                pipe_width = frm.shape[1] - i[0]  # Width available on the frame
            else:
                pipe_width = constants.PIPE_WIDTH

            if i[0] >= frm.shape[1] or pipe_width <= 0:
                continue

            upper_height = i[1]
            lower_height = self.height - i[2]

            self.place_image_on_frame(frm, self.building_image, i[0], 0, pipe_width, upper_height, index)
            self.place_image_on_frame(frm, self.building_image, i[0], i[2], pipe_width, lower_height, index)




    def update(self):
        for i in self.pipes:
            i[0] -= constants.SPEED  # Move the pipe to the left
            if i[0] + constants.PIPE_WIDTH < 0:  # If the right edge of the pipe is off the screen to the left
                self.pipes.remove(i)
            elif i[0] + constants.PIPE_WIDTH < self.width / 2 and not i[3]:  # Check if the pipe has just passed the midpoint and hasn't been scored yet
                i[3] = True  # Mark the pipe as passed to avoid multiple score increments for the same pipe
                self.points += 1  # Increment the score

    
    def check(self, index_pt):
        for i in self.pipes:
            # Directly use the width as drawn, assuming i[4] exists and is accurate
            effective_width = i[4] if len(i) > 4 else constants.PIPE_WIDTH

            # Check collision based on x-coordinate
            if i[0] <= index_pt[0] <= i[0] + effective_width:
                # Check collision based on y-coordinate
                if index_pt[1] <= i[1] or index_pt[1] >= i[2]:
                    return True  # Collision detected
        return False  # No collision detected
    
    
    def draw_image(self, frame, image_path, nose_tip_pixel, width, height):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (width, height))

        image_height, image_width, _ = image.shape

        # Calculate the position to place the image (adjust as needed for alignment)
        x_offset = nose_tip_pixel[0] - int(image_width / 2)
        y_offset = nose_tip_pixel[1] - int(image_height / 2)

        # Check if the image placement goes outside the frame boundaries
        if x_offset < 0 or y_offset < 0 or x_offset + image_width > frame.shape[1] or y_offset + image_height > frame.shape[0]:
            # Skip drawing if the image would go out of bounds
            return

        # If the image has an alpha channel, perform blending with handling transparency
        if image.shape[2] == 4:
            alpha_s = image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                frame[y_offset:y_offset+image_height, x_offset:x_offset+image_width, c] = \
                    alpha_s * image[:, :, c] + alpha_l * frame[y_offset:y_offset+image_height, x_offset:x_offset+image_width, c]
        else:
            # For images without an alpha channel, directly overlay the image
            frame[y_offset:y_offset+image_height, x_offset:x_offset+image_width] = image[:, :, :3]

    
#     def draw_debug_collision_boxes(self, frm):
#         for i in self.pipes:
#             effective_width = i[4] if len(i) > 4 else constants.PIPE_WIDTH
#             # Draw a rectangle around the pipe's collision area
#             cv2.rectangle(frm, (i[0], i[1]), (i[0] + effective_width, i[2]), (0, 255, 0), 2)




    

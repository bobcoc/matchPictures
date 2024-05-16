import cv2
import numpy as np
import os

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    return binary

def close_edges(binary_image):
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return closed

def find_largest_rectangles(contours):
    rectangles = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            rectangles.append((area, approx))
    rectangles = sorted(rectangles, key=lambda x: x[0], reverse=True)
    if not rectangles:
        return []
    max_area = rectangles[0][0]
    largest_rectangles = [rect[1] for rect in rectangles if rect[0] == max_area]
    return largest_rectangles

def detect_rectangles(image):
    binary = preprocess_image(image)
    closed = close_edges(binary)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = find_largest_rectangles(contours)
    return rectangles

def get_ordered_points(rectangle):
    rect = np.zeros((4, 2), dtype="float32")
    s = rectangle.sum(axis=1)
    rect[0] = rectangle[np.argmin(s)]
    rect[2] = rectangle[np.argmax(s)]
    diff = np.diff(rectangle, axis=1)
    rect[1] = rectangle[np.argmin(diff)]
    rect[3] = rectangle[np.argmax(diff)]
    return rect

def rectify_image(template_img, student_img):
    template_rectangles = detect_rectangles(template_img)
    student_rectangles = detect_rectangles(student_img)

    if template_rectangles and student_rectangles:
        if len(template_rectangles) != len(student_rectangles):
            print('Number of rectangles in template and student images do not match.')
            return None

        for template_pts, student_pts in zip(template_rectangles, student_rectangles):
            template_pts = template_pts.reshape(4, 2)
            student_pts = student_pts.reshape(4, 2)

            rect_template = get_ordered_points(template_pts)
            rect_student = get_ordered_points(student_pts)

            M = cv2.getPerspectiveTransform(rect_student, rect_template)
            height, width, _ = template_img.shape
            warped = cv2.warpPerspective(student_img, M, (width, height))

            mask = np.zeros_like(template_img)
            cv2.fillConvexPoly(mask, np.int32(rect_template), (255, 255, 255))
            warped = cv2.bitwise_and(warped, mask)
            template_img = cv2.bitwise_and(template_img, cv2.bitwise_not(mask))
            template_img = cv2.add(template_img, warped)

        return template_img
    else:
        print('Rectangle not detected in one or both images.')
        return None

def process_directory(template_path, src_dir, des_dir):
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    template_img = cv2.imread(template_path)

    for filename in os.listdir(src_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            student_img_path = os.path.join(src_dir, filename)
            student_img = cv2.imread(student_img_path)
            if student_img is not None:
                aligned_img = rectify_image(template_img, student_img)
                if aligned_img is not None:
                    output_path = os.path.join(des_dir, filename)
                    cv2.imwrite(output_path, aligned_img)
                    print(f'Aligned image saved to {output_path}')
            else:
                print(f'Failed to read {student_img_path}')

# 使用示例
template_image_path ='c:/c/x/template.jpg'
source_directory = 'c:/c/x/src'
destination_directory = 'c:/c/x/des'
process_directory(template_image_path, source_directory, destination_directory)

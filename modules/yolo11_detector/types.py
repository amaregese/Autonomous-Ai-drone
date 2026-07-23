class Detection:
    def __init__(self, left, top, right, bottom, class_id, class_name, confidence):
        self.Left = left
        self.Top = top
        self.Right = right
        self.Bottom = bottom
        self.Center = ((left + right) // 2, (top + bottom) // 2)
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.is_selected = False
        self.width = max(0, right - left)
        self.height = max(0, bottom - top)
        self.area = self.width * self.height
        self.aspect_ratio = self.width / self.height if self.height > 0 else 1.0

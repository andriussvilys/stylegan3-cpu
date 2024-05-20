import sys
from typing import List
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QPushButton, QWidget
from generateImage import Generator
from save_latents.image_display import ImageGrid, GeneratedImage

class MainWindow(QMainWindow):
    def __init__(self, w_count, model_pkl):
        super().__init__()
        self.G = Generator(model_pkl)
        self.setWindowTitle("Select images and save latents to JSON")

        self.selected_images = []

        self.image_viewer:ImageGrid = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.w_count = w_count
        self.latent_vectors = np.random.randn(w_count, 512)
        self.image_data:List[GeneratedImage] = self.load_images(self.latent_vectors)

        self.setup_ui()

        self.pca_windows = []


    def setup_ui(self):

        self.image_viewer = ImageGrid(self.image_data)
        self.main_layout.addWidget(self.image_viewer.main_widget, 1)

        load_more_button = QPushButton(f"Load {self.w_count} more")
        load_more_button.clicked.connect(lambda checked: self.load_more())
        self.image_viewer.add_button(load_more_button)

    def load_images(self, latent_vectors):
        images = []
        for w in latent_vectors:
            images.append(GeneratedImage(w, self.G.generate(w.reshape(1, -1))))

        return images
    
    def load_more(self):
        newLatents =  np.random.randn(self.w_count, 512)
        self.latent_vectors = self.latent_vectors + newLatents
        new_images = self.load_images(newLatents)
        self.image_viewer.update_display(self.image_viewer.image_data + new_images)


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_count", type=int, default=2,
                        help="Number of latent vectors")
    parser.add_argument("--model_pkl", type=str, default="cakes.pkl",
                        help="Model pikl file")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(w_count=args.w_count, model_pkl=args.model_pkl)
    window.setGeometry(100, 100, 600, 400)
    window.show()

    sys.exit(app.exec_())

# main()

if __name__ == "__main__":
    main()
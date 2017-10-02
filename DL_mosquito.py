import sys

from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import tensorflow as tf

class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        # self.imageLabel = QLabel()
        # self.imageLabel.setBackgroundRole(QPalette.Base)
        # self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        # self.imageLabel.setScaledContents(True)
        #
        # self.scrollArea = QScrollArea()
        # self.scrollArea.setBackgroundRole(QPalette.Dark)
        # self.scrollArea.setWidget(self.imageLabel)
        # self.setCentralWidget(self.scrollArea)

        openImage = QPushButton('Classify Image', self)
        openImage.move(10, 60)
        openImage.clicked.connect(self.open)

        self.label = QLabel("Image",self)
        self.label.setGeometry(70,80,128,128)
        self.label.move(50, 150)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Image Viewer")
        self.resize(1000, 800)



    def open(self):


        filenames = QFileDialog.getOpenFileNames(None, 'Open Folder', './')[0]
        #print(filenames[1])
        x=0
        for x in range(0,len(filenames)):
            print(filenames[x])
            image_data = load_image(filenames[x])
            run_graph(image_data, labels, 'DecodeJpeg/contents:0', 'final_result:0', 5)

            pixmap = QPixmap(filenames[x])
            pixmap.scaledToWidth(128)
            pixmap.scaledToHeight(128)
            self.label.setPixmap(pixmap)


    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                               triggered=self.open)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)


        self.menuBar().addMenu(self.fileMenu)



def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()

def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return 0


if __name__ == '__main__':

    import sys
    label_path = str("C://Users/plama/Desktop/images_test/retrained_labels.txt")
    graph_path = str("C://Users/plama/Desktop/images_test/retrained_graph.pb")
    labels = load_labels(label_path)
    load_graph(graph_path)



    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
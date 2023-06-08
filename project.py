from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableView, QPushButton, QVBoxLayout, QWidget, QLineEdit, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import pandas as pd
from data_processor import DataProcessor

class MainWindow(QMainWindow):
    rows_deleted = pyqtSignal() 
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Talent search based on embeddings")
        self.setGeometry(100, 100, 1000, 600)

        self.table_view = QTableView(self)
        self.delete_button = QPushButton("Delete Selected Rows", self)
        self.delete_button.setEnabled(False)
        self.process_button = QPushButton("Preprocess", self)
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Enter the sentence to check the similarity")
        self.calculate_button = QPushButton("Calculate the ranking", self)
        self.calculate_button.setEnabled(False)
        self.calculate_sbert_button = QPushButton("Calculate with SBERT", self)
        self.calculate_sbert_button.setEnabled(False)
        self.reset_button = QPushButton("Reset", self)
        self.reset_button.setEnabled(False)

        self.delete_button.clicked.connect(self.delete_selected_rows)
        self.process_button.clicked.connect(self.preprocess_data)
        self.calculate_button.clicked.connect(self.calculate_ranking)
        self.calculate_sbert_button.clicked.connect(self.calculate_ranking_sbert)
        self.reset_button.clicked.connect(self.reset_table)

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addWidget(self.delete_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.input_field)
        layout.addWidget(self.calculate_button)
        layout.addWidget(self.calculate_sbert_button)
        layout.addWidget(self.reset_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


        self.df = self.load_csv()  # Initialize df as an instance variable
        self.original_df = None  # Store the original DataFrame
        self.rows_deleted.connect(self.calculate_ranking)
    def load_csv(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("CSV files (*.csv)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            if filenames:
                filename = filenames[0]
                try:
                    self.df = pd.read_csv(filename)
                    columns_to_drop = ['fit', 'location']  # Specify the columns to be dropped
                    self.df = self.df.drop(columns=columns_to_drop)  # Drop the specified columns
                    self.display_dataframe(self.df)

                except pd.errors.EmptyDataError:
                    self.display_error_message("CSV file is empty")
                except pd.errors.ParserError:
                    self.display_error_message("Error parsing CSV file")
        return self.df
    def display_dataframe(self, data):
        model = QStandardItemModel(self.table_view)

        if isinstance(data, pd.DataFrame):
            # Display original dataframe
            for row in range(data.shape[0]):
                item = QStandardItem()
                item.setCheckable(True)
                item.setData(Qt.Unchecked, Qt.CheckStateRole)
                model.setItem(row, 0, item)

            column_labels = [""] + [str(column) for column in data.columns.tolist()]

            model.setHorizontalHeaderLabels(column_labels)

            for row in range(data.shape[0]):
                for column in range(data.shape[1]):
                    item = QStandardItem(str(data.iat[row, column]))
                    item.setTextAlignment(Qt.AlignCenter)
                    model.setItem(row, column + 1, item)

            self.column_names = data.columns.tolist()  # Store column names

        elif isinstance(data, list):
            # Display similarity scores
            column_labels = ["Similarity Score"]
            model.setHorizontalHeaderLabels(column_labels)

            for row, score in enumerate(data):
                item = QStandardItem(str(score))
                item.setTextAlignment(Qt.AlignCenter)
                model.setItem(row, 0, item)

        self.table_view.setModel(model)
        self.table_view.setColumnWidth(0, 50)
        self.table_view.resizeColumnsToContents()
        self.table_view.resizeRowsToContents()
        

    
    def calculate_ranking(self):
        input_sentence = self.input_field.text()
        selected_column = 2  # Assuming the column index is 2 (index starts from 0)
        if input_sentence:
            if self.df is not None and not self.df.empty:  # Check if df is available and not empty
                try:
                    data_processor = DataProcessor(self.df)
                    similarity_scores = data_processor.calculate_similarity(input_sentence)
                    # Perform further ranking calculations with similarity_scores
                    self.display_dataframe(similarity_scores)
                except Exception as e:
                    print("An error occurred during similarity calculation:", str(e))
            else:
                print("No data available. Please preprocess first.")
        else:
            QMessageBox.warning(self, "Input Sentence Empty", "Please enter a sentence to check similarity.")

    def calculate_ranking_sbert(self):
        input_sentence = self.input_field.text()
        selected_column = 2  # Assuming the column index is 2 (index starts from 0)
        if input_sentence:
            if self.df is not None and not self.df.empty:  # Check if df is available and not empty
                try:
                    data_processor = DataProcessor(self.df)
                    similarity_scores = data_processor.calculate_similarity_bert(input_sentence)
                    # Perform further ranking calculations with similarity_scores
                    self.display_dataframe(similarity_scores)
                except Exception as e:
                    print("An error occurred during similarity calculation:", str(e))
            else:
                print("No data available. Please preprocess first.")
        else:
            QMessageBox.warning(self, "Input Sentence Empty", "Please enter a sentence to check similarity.")



    def delete_selected_rows(self):
        model = self.table_view.model()

        selected_rows = []
        for row in range(model.rowCount() - 1, -1, -1):
            index = model.index(row, 0)
            checked = model.data(index, Qt.CheckStateRole)

            if checked == Qt.Checked:
                selected_rows.append(row)
                id_value = model.index(row, 1).data()  # Assuming the 'id' column index is 1
                self.df = self.df[self.df['id'] != id_value].reset_index(drop=True)
                model.removeRow(row)

        if selected_rows:
            self.rows_deleted.emit()





    def display_error_message(self, message):
        self.table_view.setModel(None)
        self.setWindowTitle("Error")
        self.table_view.clear()
        self.table_view.setAlternatingRowColors(False)
        self.table_view.setStyleSheet("QTableView {background-color: #ffcccc;}")
        self.table_view.setTextElideMode(Qt.ElideRight)
        self.table_view.setTextElideMode(Qt.ElideNone)
        self.table_view.setGridStyle(Qt.NoPen)
        self.table_view.setShowGrid(False)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.horizontalHeader().setVisible(False)
        self.table_view.horizontalHeader().setSectionResizeMode(QTableView.Stretch)
        self.table_view.setWordWrap(True)
        self.table_view.setSpan(0, 0, 1, self.table_view.model().columnCount())
        self.table_view.model().setItem(0, 0, QStandardItem(message))



    def preprocess_data(self):
        model = self.table_view.model()
        data = []

        for row in range(model.rowCount()):
            row_data = []
            for column in range(model.columnCount()):
                index = model.index(row, column)
                item = model.data(index)
                row_data.append(item)
            data.append(row_data)
        self.original_df = pd.DataFrame(data)  # Store the original DataFrame
        self.df = self.original_df.copy()  # Make a copy for preprocessing

        if self.column_names:
            self.df = self.df.drop(self.df.columns[0], axis=1)
            self.df.columns = self.column_names  # Set the original column name

        self.df = self.df.drop_duplicates(subset=['job_title', 'connection'])
        data_processor = DataProcessor(self.df)
        self.df[self.df.columns[1]] = self.df[self.df.columns[1]].apply(data_processor.preprocess)

        if self.column_names:
            self.display_dataframe(self.df)
            self.calculate_button.setEnabled(True)
            self.calculate_sbert_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            self.delete_button.setEnabled(True)
        else:
            self.column_names = self.df.columns.tolist()
            self.display_dataframe(self.df)
            self.reset_button.setEnabled(True)

    def reset_table(self):
        self.df = self.original_df.copy()  # Restore the original DataFrame
        self.df = self.df.drop(self.df.columns[0], axis=1)
        self.df.columns = ['id','job_title', 'connection']  # Update the column names
        self.display_dataframe(self.df)
        self.calculate_button.setEnabled(False)
        self.calculate_sbert_button.setEnabled(False)
        self.reset_button.setEnabled(False)




if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

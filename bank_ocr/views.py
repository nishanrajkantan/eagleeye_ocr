from django.db import models
from django.shortcuts import redirect, render
from django.views.generic import ListView, CreateView, TemplateView
from django.urls import reverse, reverse_lazy
from .forms import PostForm
from .models import Post
from django.core.files.base import ContentFile
from django.core.files import File


# Create your views here.
class HomePageView(ListView):
    model = Post
    template_name = 'home.html'

class CreatePostView(CreateView):
    model = Post
    form_class = PostForm
    template_name = 'post.html'
    success_url = reverse_lazy('result')

    def form_valid(self, form):
        return super().form_valid(form)

class ResultView(TemplateView):
    template_name = "result.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        obj = Post.objects.latest('id')

        # ---------------------------------------------------------------------------------------------------------------------
        # Script 13. test.py
        # ---------------------------------------------------------------------------------------------------------------------

        import pikepdf
        from PyPDF2 import PdfFileReader
        from tqdm import tqdm
        from pdf2image import convert_from_path
        import pandas as pd
        import subprocess
        import os
        from inferenceutils import label_map_util, load_image_into_numpy_array, run_inference_for_single_image
        import tensorflow as tf

        zipped_model = 'inferenceutils.py'
        if not os.path.exists(zipped_model):
            subprocess.call(
                'powershell -Command "Invoke-WebRequest https://raw.githubusercontent.com/hugozanini/object-detection/master/inferenceutils.py -OutFile .\\inferenceutils.py"')

        # ---------------------------------------------------------------------------------------------------------------------
        # CIMB Model
        # ---------------------------------------------------------------------------------------------------------------------

        zipped_model = 'cimb_model\\saved_model'
        if not os.path.exists(zipped_model):
            subprocess.call(
                "powershell Expand-Archive -Path 'cimb_model\\saved_model.zip' -DestinationPath 'cimb_model\\saved_model\\'")

        labelmap_path = 'cimb_model\\saved_model\\saved_model\\label_map.pbtxt'

        category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
        tf.keras.backend.clear_session()
        model = tf.saved_model.load(f'cimb_model\\saved_model\\saved_model')


        process_path = '..\\media\\processing_images\\cimb\\'
        input_path1 = '..\\media\\raw_dataset\\cimb\\'
        output_path = '..\\media\\predicted_coordinates_cimb.csv'

        if not os.path.exists(process_path):
            subprocess.call("powershell mkdir " + process_path)

        for i, pdf in tqdm(enumerate(os.listdir(input_path1))):
            file = pikepdf.open(input_path1 + pdf, allow_overwriting_input=True)
            file.save(input_path1 + pdf)

        for i, pdf in tqdm(enumerate(os.listdir(input_path1))):
            input = PdfFileReader(open(input_path1 + pdf, 'rb'))
            width = input.getPage(0).mediaBox[2]
            height = input.getPage(0).mediaBox[3]
            images = convert_from_path(input_path1 + pdf, size=(width, height))
            for i, image in enumerate(images):
                image.save(process_path + pdf[:-4] + '_page_' + str(i) + '.jpg', 'JPEG')

        # Getting images to test
        images = os.listdir(process_path)

        # for multiple images, use this code
        rows = []
        for image_name in images:

            image_np = load_image_into_numpy_array(process_path + image_name)
            output_dict = run_inference_for_single_image(model, image_np)

            # store boxes in dataframe!
            cut_off_scores = len(list(filter(lambda x: x >= 0.1, output_dict['detection_scores'])))

            for j in range(cut_off_scores):
                name = image_name
                scores = output_dict['detection_scores'][j]
                classes = output_dict['detection_classes'][j]
                for i in range(1, len(category_index) + 1):
                    if output_dict['detection_classes'][j] == category_index[i]['id']:
                        classes = category_index[i]['name']
                ymin = output_dict['detection_boxes'][j][0]
                xmin = output_dict['detection_boxes'][j][1]
                ymax = output_dict['detection_boxes'][j][2]
                xmax = output_dict['detection_boxes'][j][3]

                row = list([name, scores, classes, ymin, xmin, ymax, xmax])
                rows.append(row)

        final_df = pd.DataFrame(rows, columns=['Image', 'Scores', 'Classes', 'ymin', 'xmin', 'ymax', 'xmax'])
        final_df.to_csv(output_path, index=False)

        # ---------------------------------------------------------------------------------------------------------------------
        # Maybank Model
        # ---------------------------------------------------------------------------------------------------------------------

        zipped_model = 'maybank_model\\saved_model'
        if not os.path.exists(zipped_model):
            subprocess.call(
                "powershell Expand-Archive -Path 'test\\saved_model.zip' -DestinationPath 'maybank_model\\saved_model\\'")

        output_directory = 'inference_graph'
        labelmap_path = 'maybank_model\\saved_model\\saved_model\\label_map.pbtxt'

        category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
        tf.keras.backend.clear_session()
        model = tf.saved_model.load(f'maybank_model\\saved_model\\saved_model')

        process_path = '..\\media\\processing_images\\mayb\\'
        input_path1 = '..\\media\\raw_dataset\\mayb\\'
        output_path = '..\\media\\predicted_coordinates_mayb.csv'

        if not os.path.exists(process_path):
            subprocess.call("powershell mkdir " + process_path)

        for i, pdf in tqdm(enumerate(os.listdir(input_path1))):
            file = pikepdf.open(input_path1 + pdf, allow_overwriting_input=True)
            file.save(input_path1 + pdf)

        for i, pdf in tqdm(enumerate(os.listdir(input_path1))):
            input = PdfFileReader(open(input_path1 + pdf, 'rb'))
            width = input.getPage(0).mediaBox[2]
            height = input.getPage(0).mediaBox[3]
            images = convert_from_path(input_path1 + pdf, size=(width, height))
            for i, image in enumerate(images):
                image.save(process_path + pdf[:-4] + '_page_' + str(i) + '.jpg', 'JPEG')

        # Getting images to test
        images = os.listdir(process_path)

        # for multiple image, use this code
        rows = []
        for image_name in images:

            image_np = load_image_into_numpy_array(process_path + image_name)
            output_dict = run_inference_for_single_image(model, image_np)

            # store boxes in dataframe!
            cut_off_scores = len(list(filter(lambda x: x >= 0.1, output_dict['detection_scores'])))

            for j in range(cut_off_scores):
                name = image_name
                scores = output_dict['detection_scores'][j]
                classes = output_dict['detection_classes'][j]
                for i in range(1, len(category_index) + 1):
                    if output_dict['detection_classes'][j] == category_index[i]['id']:
                        classes = category_index[i]['name']
                ymin = output_dict['detection_boxes'][j][0]
                xmin = output_dict['detection_boxes'][j][1]
                ymax = output_dict['detection_boxes'][j][2]
                xmax = output_dict['detection_boxes'][j][3]

                row = list([name, scores, classes, ymin, xmin, ymax, xmax])
                rows.append(row)

        final_df = pd.DataFrame(rows, columns=['Image', 'Scores', 'Classes', 'ymin', 'xmin', 'ymax', 'xmax'])
        final_df.to_csv(output_path, index=False)

        # ---------------------------------------------------------------------------------------------------------------------
        # Script 14. pdf_extract_table.py
        # ---------------------------------------------------------------------------------------------------------------------

        import os
        import re
        import subprocess
        import camelot
        import pandas as pd
        import pikepdf
        from tqdm import tqdm

        input_path_cimb = '..\\media\\raw_dataset\\cimb\\'
        input_path_mayb = '..\\media\\raw_dataset\\mayb\\'

        cimb_files = sorted(os.listdir(input_path_cimb))
        mayb_files = sorted(os.listdir(input_path_mayb))

        output_path_cimb = '..\\media\\transaction_output\\cimb\\'
        output_path_mayb = '..\\media\\transaction_output\\mayb\\'

        if not os.path.exists(output_path_cimb):
            subprocess.call("powershell mkdir " + output_path_cimb)

        if not os.path.exists(output_path_mayb):
            subprocess.call("powershell mkdir " + output_path_mayb)

        for i, pdf in tqdm(enumerate(cimb_files)):
            file = pikepdf.open(input_path_cimb + pdf, allow_overwriting_input=True)
            file.save(input_path_cimb + pdf)

        for i, pdf in tqdm(enumerate(mayb_files)):
            file = pikepdf.open(input_path_mayb + pdf, allow_overwriting_input=True)
            file.save(input_path_mayb + pdf)

        for i, pdf in tqdm(enumerate(cimb_files)):
            table = camelot.read_pdf(input_path_cimb + pdf, pages='all', flavor='lattice')
            tables = pd.DataFrame()
            with pd.ExcelWriter(output_path_cimb + 'transaction_' + pdf[:-4] + '.xlsx', engine='xlsxwriter') as writer:
                tables = pd.DataFrame()
                for x in range(len(table)):
                    if x == 0:
                        tables = tables.append(table[0].df)
                    elif table[0].df[0][0] == table[x].df[0][0]:
                        tables = tables.append(table[x].df.iloc[1:])
                tables = tables.reset_index().drop(columns=['index'], axis=1)

                columns = tables[0][0].split("\n")
                columns = columns[:7]

                dates = []
                descriptions = []
                refs = []
                withdraws = []
                deposits = []
                taxes = []
                balances = []
                for i in range(2, len(tables)):

                    # Get date & description
                    # ----------------------------------------
                    date = tables[0][i][:10]
                    description = tables[0][i][12:]
                    add_description = tables[1][i]
                    ref = tables[2][i]
                    # ----------------------------------------

                    # Get withdrawal (debit)
                    # ----------------------------------------
                    withdraw = re.sub(',', '', tables[3][i])
                    if withdraw == '':
                        withdraw = 0
                    else:
                        withdraw = float(withdraw)
                    # ----------------------------------------

                    # Get deposit (credit)
                    # ----------------------------------------
                    deposit = re.sub(',', '', tables[4][i])
                    if deposit == '':
                        deposit = 0
                    else:
                        deposit = float(deposit)
                    # ----------------------------------------

                    # Get tax
                    # ----------------------------------------
                    tax = re.sub(',', '', tables[5][i])
                    if tax == '':
                        tax = 0
                    else:
                        tax = float(tax)
                    # ----------------------------------------

                    # Get account balance
                    # ----------------------------------------
                    balance = re.sub(',', '', tables[6][i])
                    if balance == '':
                        balance = 0
                    else:
                        balance = float(balance)
                    # ----------------------------------------

                    # Append all lists
                    # ----------------------------------------
                    dates.append(date)
                    descriptions.append(description + '\n' + add_description)
                    refs.append(ref)
                    withdraws.append(withdraw)
                    deposits.append(deposit)
                    taxes.append(tax)
                    balances.append(balance)
                    # ----------------------------------------

                transactions_df = pd.DataFrame(
                    list([dates, descriptions, refs, withdraws, deposits, taxes, balances])).T
                transactions_df.columns = columns
                transactions_df.to_excel(writer, sheet_name='transactions', index=False)

        for i, pdf in tqdm(enumerate(mayb_files)):

            table = camelot.read_pdf(input_path_mayb + pdf, pages='all', flavor='stream')
            tables = pd.DataFrame()
            with pd.ExcelWriter(output_path_mayb + 'transaction_' + pdf[:-4] + '.xlsx', engine='xlsxwriter') as writer:

                table = camelot.read_pdf(input_path_mayb + pdf, pages='all', flavor='lattice')

                tables = pd.DataFrame()
                for x in range(len(table)):
                    if x == 0:
                        tables = tables.append(table[0].df)
                    elif x != 0:
                        tables = tables.append(table[x].df.iloc[1:])
                tables = tables.reset_index().drop(columns=['index'], axis=1)
                tables1 = tables.iloc[1:].reset_index().drop(columns=['index'], axis=1)

                for i in range(len(tables.columns)):
                    multi_row_column = tables.iloc[:, i].to_list()
                    tables1[0][i] = '\n'.join(multi_row_column)

                tables1 = tables1.iloc[:1]
                tables1.columns = tables.iloc[0]

                columns = tables.iloc[0, :]

                dates = []
                descs = []
                trans = []
                balances = []

                for i in range(1, len(tables)):
                    value = tables.iloc[i, :]

                    date = value[0].split("\n")
                    description = value[1].split("\n")
                    transaction = value[2].split("\n")
                    balance = value[3].split("\n")

                    dates.extend(date)
                    descs.extend(description)
                    trans.extend(transaction)
                    balances.extend(balance)

                descriptions = []
                for i in range(len(descs)):
                    if re.search("   ", descs[i]) != None:
                        descriptions[-1] = descriptions[-1] + descs[i]
                    else:
                        descriptions.append(descs[i])
                descriptions = descriptions[1:]

                for i in range(len(descriptions)):
                    try:
                        if (descriptions[i][:22] == 'PAYMENT VIA MYDEBIT   ') and (
                                len(descriptions[i].split('   ')) == 3):
                            descriptions[i] = descriptions[i] + ('   ') + descriptions[i + 1]
                            del descriptions[i + 1]
                        elif (descriptions[i][:19] == 'FUND TRANSFER TO A/') and (
                                len(descriptions[i].split('   ')) == 3):
                            descriptions[i] = descriptions[i] + ('   ') + descriptions[i + 1]
                            del descriptions[i + 1]

                        elif (descriptions[i][:20] == 'FPX PAYMENT FR A/   ') and (
                                len(descriptions[i].split('   ')) == 3) or (len(descriptions[i].split('   ')) == 2):
                            descriptions[i] = descriptions[i] + ('   ') + descriptions[i + 1]
                            del descriptions[i + 1]

                        elif (descriptions[i] == 'ENDING BALANCE :'):
                            del descriptions[i:]
                    except:
                        pass

                trans = trans[:-3]
                balances = balances[1:]
                df = pd.DataFrame((dates, descriptions, trans, balances)).T
                df.columns = columns
                df = df.dropna()

                df['CREDIT'] = 0
                df['DEBIT'] = 0

                for i in range(len(df)):
                    if df[df.columns[2]][i].find('+') != -1:
                        df['CREDIT'][i] = re.sub(r'[,+-]+', r'', df[df.columns[2]][i])
                        df['DEBIT'][i] = 0
                        df[df.columns[3]][i] = re.sub(r',', r'', df[df.columns[3]][i])

                    elif df[df.columns[2]][i].find('-') != -1:
                        df['DEBIT'][i] = re.sub(r'[,+-]+', r'', df[df.columns[2]][i])
                        df['CREDIT'][i] = 0
                        df[df.columns[3]][i] = re.sub(r',', r'', df[df.columns[3]][i])

                df[df.columns[3]] = df[df.columns[3]].astype('float')
                df['DEBIT'] = df['DEBIT'].astype('float')
                df['CREDIT'] = df['CREDIT'].astype('float')

                df.to_excel(writer, sheet_name='transactions', index=False)

        # ---------------------------------------------------------------------------------------------------------------------
        # Script 15. extract_metadata.py
        # ---------------------------------------------------------------------------------------------------------------------

        import os
        import subprocess
        import pytesseract
        import cv2
        import glob
        from PIL import Image
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        from pytesseract import Output

        # --------------------------------------------------------------------------------------------------------------
        # Prediction on CIMB transactions
        # --------------------------------------------------------------------------------------------------------------

        coordinates_df = pd.read_csv('..\\media\\predicted_coordinates_cimb.csv')
        input_path = '..\\media\\processing_images\\cimb\\'
        process_path = '..\\media\\cropped_images\\cimb\\'
        output_path = '..\\media\\financial_output\\cimb\\'
        excel_file_path = '..\\media\\transaction_output\\cimb\\'
        excel_files = sorted(os.listdir(excel_file_path))

        # --------------------------------------------------------------------------------------------------------------

        if not os.path.exists(process_path):
            subprocess.call("powershell mkdir " + process_path)

        if not os.path.exists(output_path):
            subprocess.call("powershell mkdir " + output_path)

        images = glob.glob1(input_path, "*.jpg")
        tessdata_dir_config = r'-l "eng+msa" --tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata" --psm 6'
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        # --------------------------------------------------------------------------------------------------------------

        rows = []
        for i in tqdm(range(len(coordinates_df))):
            for j in range(len(images)):
                if coordinates_df['Image'][i] == images[j]:
                    image = Image.open(input_path + images[j])
                    results = pytesseract.image_to_data(image, config=tessdata_dir_config, output_type=Output.DICT)

                    # (This is not mandatory)
                    width, height = image.size

                    # Setting the points for cropped image
                    left = coordinates_df['xmin'][i] * width
                    top = coordinates_df['ymin'][i] * height
                    right = coordinates_df['xmax'][i] * width
                    bottom = coordinates_df['ymax'][i] * height

                    # Cropped image of above dimension
                    # (It will not change orginal image)
                    im1 = image.crop((left, top, right, bottom))
                    im1 = im1.resize((im1.size[0] * 5, im1.size[1] * 5), resample=5)

                    # # Shows the image in image viewer
                    # im1.show()
                    im1.save(process_path + images[j])
                    text = pytesseract.image_to_string(im1, lang='eng+msa', config=tessdata_dir_config)

                    row = list([images[j], coordinates_df['Classes'][i], text])
                    rows.append(row)

        raw_info = pd.DataFrame(rows, columns=['Image', 'Key', 'Value'])

        # --------------------------------------------------------------------------------------------------------------

        for i, excel in tqdm(enumerate(excel_files)):
            output_df = pd.read_excel(excel_file_path + excel)
            statement_name1 = excel[12:-5]
            with pd.ExcelWriter(output_path + 'output_' + excel[12:-5] + '.xlsx', engine='xlsxwriter') as writer:
                rows = []
                for j in range(len(raw_info)):
                    statement_name2 = raw_info['Image'][j][:15]
                    if (statement_name1 == statement_name2) and (raw_info['Key'][j] != 'transactions'):
                        row = list(raw_info.iloc[j][1:])
                        rows.append(row)
                        df = pd.DataFrame(rows, columns=['Key', 'Value'])
                        df = df.drop_duplicates()
                output_df.to_excel(writer, sheet_name='transactions', index=False)
                df.to_excel(writer, sheet_name='metadata', index=False)

        # --------------------------------------------------------------------------------------------------------------
        # Prediction on Maybank transactions
        # --------------------------------------------------------------------------------------------------------------

        coordinates_df2 = pd.read_csv('..\\media\\predicted_coordinates_mayb.csv')
        input_path2 = '..\\media\\processing_images\\mayb\\'
        process_path2 = '..\\media\\cropped_images\\mayb\\'
        output_path2 = '..\\media\\financial_output\\mayb\\'
        excel_file_path2 = '..\\media\\transaction_output\\mayb\\'
        excel_files2 = sorted(os.listdir(excel_file_path2))

        # --------------------------------------------------------------------------------------------------------------

        if not os.path.exists(process_path2):
            subprocess.call("powershell mkdir " + process_path2)

        if not os.path.exists(output_path2):
            subprocess.call("powershell mkdir " + output_path2)

        images2 = glob.glob1(input_path2, "*.jpg")
        tessdata_dir_config = r'-l "eng+msa" --tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata" --psm 6'
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        # --------------------------------------------------------------------------------------------------------------

        rows2 = []
        for i in tqdm(range(len(coordinates_df2))):
            for j in range(len(images2)):
                if coordinates_df2['Image'][i] == images2[j]:
                    image = Image.open(input_path2 + images2[j])
                    results = pytesseract.image_to_data(image, config=tessdata_dir_config, output_type=Output.DICT)

                    # (This is not mandatory)
                    width, height = image.size

                    # Setting the points for cropped image
                    left = coordinates_df2['xmin'][i] * width
                    top = coordinates_df2['ymin'][i] * height
                    right = coordinates_df2['xmax'][i] * width
                    bottom = coordinates_df2['ymax'][i] * height

                    # Cropped image of above dimension
                    # (It will not change orginal image)
                    im1 = image.crop((left, top, right, bottom))
                    im1 = im1.resize((im1.size[0] * 5, im1.size[1] * 5), resample=5)

                    # # Shows the image in image viewer
                    # im1.show()
                    im1.save(process_path2 + images2[j])
                    text = pytesseract.image_to_string(im1, lang='eng+msa', config=tessdata_dir_config)

                    row = list([images2[j], coordinates_df2['Classes'][i], text])
                    rows2.append(row)

        raw_info2 = pd.DataFrame(rows2, columns=['Image', 'Key', 'Value'])

        # --------------------------------------------------------------------------------------------------------------

        for i, excel in tqdm(enumerate(excel_files2)):
            output_df2 = pd.read_excel(excel_file_path2 + excel)
            statement_name1 = excel[12:-5]
            with pd.ExcelWriter(output_path2 + 'output_' + excel[12:-5] + '.xlsx', engine='xlsxwriter') as writer2:
                rows = []
                for j in range(len(raw_info2)):
                    statement_name2 = raw_info2['Image'][j][:18]
                    if (statement_name1 == statement_name2) and (raw_info2['Key'][j] != 'transactions'):
                        row = list(raw_info2.iloc[j][1:])
                        rows.append(row)
                        df2 = pd.DataFrame(rows, columns=['Key', 'Value'])
                        df2 = df2.drop_duplicates()
                output_df2.to_excel(writer2, sheet_name='transactions', index=False)
                df2.to_excel(writer2, sheet_name='metadata', index=False)

        # --------------------------------------------------------------------------------------------------------------
        # 16. generate_sender_receiver_column.py
        # --------------------------------------------------------------------------------------------------------------

        import pandas as pd
        import os
        import glob
        import re
        import subprocess
        from tqdm import tqdm

        input_path = '..\\media\\financial_output\\mayb\\'
        file_list = glob.glob1(input_path, "*.xlsx")
        output_path = '..\\media\\extract_description\\raw_files\\'

        if not os.path.exists(output_path):
            subprocess.call("powershell mkdir " + output_path)

        for j in tqdm(range(len(file_list))):

            df_transaction = pd.read_excel(input_path + file_list[j], sheet_name='transactions')
            df_metadata = pd.read_excel(input_path + file_list[j], sheet_name='metadata')
            df_metadata['Value'] = df_metadata['Value'].str.replace('\n_x000C_', '')

            senders = []
            receivers = []

            for i in tqdm(range(len(df_transaction))):
                if re.search('SVG GIRO CR',
                             df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern = re.search('SVG GIRO CR',
                                        df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern.span()[1]:]

                elif re.search('SALE DEBIT',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    entity = 'ONLINE DEBIT'
                    # entity = ''

                elif re.search('DEBIT ADVICE',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    entity = 'BANK DEDUCTION'
                    # entity = ''

                elif re.search('CASH WITHDRAWAL',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    entity = 'ATM WITHDRAWAL'
                    # entity = ''

                elif re.search('IBK FUND TFR FR A/C',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('IBK FUND TFR FR A/C',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:pattern2.span()[0]]
                    # entity = ''

                elif re.search('CLEARING CHQ DEP',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    entity = 'SALARY INCOME'
                    # entity = ''

                elif re.search('TRANSFER FROM A/C',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('TRANSFER FROM A/C',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:pattern2.span()[0]]
                    # entity = ''

                elif re.search('PAYMENT VIA MYDEBIT',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('PAYMENT VIA MYDEBIT',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:]
                    # entity = ''

                elif re.search('FUND TRANSFER TO A/',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('FUND TRANSFER TO A/',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:]
                    # entity = ''

                elif re.search('FPX PAYMENT FR A/',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    # entity_list = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i].split('   ')
                    # print(entity_list)
                    # entity = entity_list[2]
                    pattern1 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[0-9]+', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i][
                                                   pattern1.span()[1]:])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:]
                    # entity = ''

                elif re.search('FUND TRANSFER TO',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('FUND TRANSFER TO',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:]
                    # entity = ''

                elif re.search('IBK FUND TFR TO A/C',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('IBK FUND TFR TO A/C',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:pattern2.span()[0]]
                    # entity = ''

                elif re.search('PYMT FROM A/C',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('PYMT FROM A/C',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    pattern2 = re.search('[*]', df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    description_text = df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]
                    entity = description_text[pattern1.span()[1]:pattern2.span()[0]]
                    # entity = ''

                elif re.search('HIBAH PAID',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('HIBAH PAID',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    entity = 'INSURANCE HIBAH'
                    # entity = ''

                elif re.search('CASH DEPOSIT',
                               df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i]) != None:
                    pattern1 = re.search('CASH DEPOSIT',
                                         df_transaction['BUTIR URUSNIAGA\n進支項說明\nTRANSACTION DESCRIPTION'][i])
                    entity = 'ATM DEPOSIT'
                    # entity = ''

                else:
                    entity = ''

                if df_transaction['CREDIT'][i] != 0:
                    sender = str(entity)
                    receiver = df_metadata[df_metadata['Key'] == 'account_holder']['Value'].values[0]
                else:
                    sender = df_metadata[df_metadata['Key'] == 'account_holder']['Value'].values[0]
                    receiver = str(entity)

                senders.append(sender)
                receivers.append(receiver)

            df_new = pd.DataFrame({'SENDER': senders, 'RECEIVER': receivers})
            df_final = pd.concat([df_transaction, df_new], axis=1)
            df_final.to_excel(output_path + 'output2_' + file_list[j][7:], index=False)

        # --------------------------------------------------------------------------------------------------------------
        # 17. build_eagleyedb.py
        # --------------------------------------------------------------------------------------------------------------

        import pandas as pd
        import numpy as np
        import glob
        import os
        import subprocess
        from tqdm import tqdm

        # file_list = glob.glob1('C:\\Users\\DataMicron\\Desktop\\Bank_Statement_Reader\\extract_description\\raw_files\\', '*.xlsx')
        input_path = '..\\media\\extract_description\\edited_files\\'
        file_list = glob.glob1(input_path, '*.xlsx')
        output_path = '..\\media\\eagleyedb\\'

        if not os.path.exists(output_path):
            subprocess.call("powershell mkdir " + output_path)

        def flatten(list_of_list):
            return [item for sublist in list_of_list for item in sublist]

        def get_entity_type_id(datalake_df):
            entity_type_id = []
            for i in range(len(datalake_df.columns)):
                column_index_no = datalake_df.columns.get_loc(datalake_df.columns[i]) + 1
                entity_type_id.append(column_index_no)
            return entity_type_id

        def get_entity_type(datalake_df):
            datalake_df.columns = map(str.upper, datalake_df.columns)
            entity_type = datalake_df.columns.to_list()
            return entity_type

        def get_entity_desc(entity_type):
            entity_type_desc = entity_type.copy()
            return entity_type_desc

        def get_entity(datalake_df):
            entities = flatten(datalake_df.values.tolist())
            return entities

        def get_entity_id(entities):
            entity_ids = []
            for i in range(len(entities)):
                entity_id = i + 1
                entity_ids.append(entity_id)
            return entity_ids

        def get_factor(entity_id, entity_type):
            factor = len(entity_id) / len(entity_type)
            return int(factor)

        def get_factor2(entity_id):
            factor = len(entity_id)
            return int(factor)

        def get_entity_type2(entity_id, entity_type_id):
            entity_type = entity_type_id.copy()
            factor = get_factor(entity_id, entity_type)
            entity_type = entity_type_id.copy() * factor
            return entity_type

        def get_location(factor):
            locations = ['' for i in range(factor)]
            return locations

        def get_address(factor):
            addresses = ['' for i in range(factor)]
            return addresses

        def get_image(entity_type):
            images = []
            image = ''
            for i in range(len(entity_type)):
                if entity_type[i] == 1:
                    image = '/assets/images/money.png'
                elif entity_type[i] == 2:
                    image = '/assets/images/person.png'
                elif entity_type[i] == 3:
                    image = '/assets/images/person.png'
                else:
                    pass
                images.append(image)
            return images

        def get_doc_id(factor):
            doc_ids = [1 for i in range(factor)]
            return doc_ids

        def get_search_around(entity_type):
            searcharounds = []
            for i in range(len(entity_type)):
                if entity_type[i] == 1:
                    searcharound = ''
                elif entity_type[i] == 2:
                    searcharound = ''
                elif entity_type[i] == 3:
                    searcharound = ''
                searcharounds.append(searcharound)
            return searcharounds

        def get_search_around_url(factor):
            search_around_url = ['' for i in range(factor)]
            return search_around_url

        def get_event_name():
            event_name = ['send', '']
            return event_name

        def get_event_desc(event_name):
            event_desc = event_name.copy()
            return event_desc

        def get_event_id(event_name):
            event_ids = []
            for i in range(len(event_name)):
                event_id = i + 1
                event_ids.append(event_id)
            return event_ids

        def random_dates(start, end, n=1):
            start_u = start.value // 10 ** 9
            end_u = end.value // 10 ** 9
            return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

        def get_doc_id2(entity_id):
            docids = [1 for i in range(len(entity_id))]
            return docids

        def get_entity_id2(entity_id):
            entityids = entity_id.copy()
            return entityids

        def get_start_idx(entity_id):
            start_idx = [0 for i in range(len(entity_id))]
            return start_idx

        def get_end_idx(entity_id):
            end_idx = [5 for i in range(len(entity_id))]
            return end_idx

        def get_date(dates_df):
            dates_list = []
            for i in range(len(dates_df)):
                date = dates_df['TARIKH MASUK'][i]
                for j in range(3):
                    dates_list.append(date)
            return dates_list

        def get_date2(dates_df):
            dates_list = []
            for i in range(len(dates_df)):
                date = dates_df['TARIKH MASUK'][i]
                for j in range(3 - 1):
                    dates_list.append(date)
            return dates_list

        def get_entity_unique_id(docid, entity_id, start_idx, end_idx):
            df = pd.DataFrame([docid, entity_id, start_idx, end_idx]).T
            df.columns = ['docid', 'entityid', 'startidx', 'endidx']
            df['entityuniqueid'] = ''
            for i in range(len(df)):
                df['entityuniqueid'][i] = int(
                    str(df['docid'][i]) + str(df['entityid'][i]) + str(df['startidx'][i] + df['endidx'][i]))
            entityuniqueid = df['entityuniqueid'].to_list()
            return entityuniqueid

        def get_relations(entity_df, entity_mentions_df):
            merged_df = entity_df.merge(entity_mentions_df, left_on='EntityId', right_on='EntityId')

            # -----------------------------------------------------------------------------------

            # Replace entityuniqueid of similar entities
            filtered_df = merged_df[~merged_df['EntityType'].isin([6, 9, 10, 11, 12])].dropna(subset=['EntityName'])
            duplicated_df2 = filtered_df[filtered_df.duplicated(subset=['EntityName'])]
            sorted_df2 = duplicated_df2.sort_values(ascending=True, by='EntityName').reset_index(drop=True)
            for j in range(1, len(sorted_df2)):
                if sorted_df2['EntityName'][j] == sorted_df2['EntityName'][j - 1]:
                    sorted_df2['EntityUniqueId'][j] = sorted_df2['EntityUniqueId'][j - 1]
                else:
                    sorted_df2['EntityUniqueId'][j] = sorted_df2['EntityUniqueId'][j]
            for i in range(len(sorted_df2)):
                for j in range(len(merged_df)):
                    if sorted_df2['EntityName'][i] == merged_df['EntityName'][j]:
                        merged_df['EntityUniqueId'][j] = sorted_df2['EntityUniqueId'][i]

            # Split entity by entity id
            entity_type1_df = merged_df[merged_df['EntityType'] == 1].reset_index(drop=True)
            entity_type2_df = merged_df[merged_df['EntityType'] == 2].reset_index(drop=True)
            entity_type3_df = merged_df[merged_df['EntityType'] == 3].reset_index(drop=True)

            # -----------------------------------------------------------------------------------

            # Map Relations based on their entity type
            entity1ids = []
            entity2ids = []
            entity1dtypes = []
            entity2dtypes = []

            entity1d = entity_type2_df['EntityUniqueId'].tolist()
            entity1dtype = entity_type2_df['EntityType'].tolist()
            entity1ids.extend(entity1d)
            entity1dtypes.extend(entity1dtype)
            entity2d = entity_type1_df['EntityUniqueId'].tolist()
            entity2dtype = entity_type1_df['EntityType'].tolist()
            entity2ids.extend(entity2d)
            entity2dtypes.extend(entity2dtype)

            entity1d = entity_type1_df['EntityUniqueId'].tolist()
            entity1dtype = entity_type1_df['EntityType'].tolist()
            entity1ids.extend(entity1d)
            entity1dtypes.extend(entity1dtype)
            entity2d = entity_type3_df['EntityUniqueId'].tolist()
            entity2dtype = entity_type3_df['EntityType'].tolist()
            entity2ids.extend(entity2d)
            entity2dtypes.extend(entity2dtype)

            # -----------------------------------------------------------------------------------

            return entity1ids, entity2ids, entity1dtypes, entity2dtypes, merged_df

        def get_columns():
            columns = ['DocId', 'Entity1Id', 'Entity2Id', 'Entity1Type', 'Entity2Type', 'Date']
            return columns

        def get_columns2():
            columns = ['DocId', 'EventId', 'Entity1Id', 'Entity2Id', 'Date']
            return columns

        def get_event_id2(event_mentions_df):
            event_ids = []
            event_id = ''
            for i in range(len(event_mentions_df)):
                if event_mentions_df['Entity2Type'][i] == 2:
                    event_id = 1
                elif event_mentions_df['Entity2Type'][i] == 3:
                    event_id = 1
                else:
                    event_id = 1
                event_ids.append(event_id)
            return event_ids

        def clean_na(event_mentions_df2, merged_df):
            column_name = event_mentions_df2.columns.to_list()
            working_df = event_mentions_df2.merge(merged_df, left_on='Entity2Id', right_on='EntityUniqueId',
                                                  suffixes=('', '_x'))
            working_df = working_df.dropna(subset=['EntityName'])
            working_df = working_df[working_df['EntityName'] != ''].reset_index(drop=True)
            working_df2 = working_df[column_name]
            return working_df2, working_df

        def get_relations3(connected_entity_df, event_mentions_df2):
            column_name = event_mentions_df2.columns.to_list()
            connected_entity_df = connected_entity_df[connected_entity_df.duplicated(subset='EntityName', keep=False)]
            connected_entity_df['Entitiy1Id_copy'] = connected_entity_df['Entity1Id']
            connected_entity_df['Entitiy2Id_copy'] = connected_entity_df['Entity2Id']
            connected_entity_df['Entity1Id'] = connected_entity_df['Entitiy2Id_copy']
            connected_entity_df['Entity2Id'] = connected_entity_df['Entitiy1Id_copy']
            connected_entity_df['EventId'] = 2
            connected_entity_df = connected_entity_df[column_name]
            event_mentions_df2 = event_mentions_df2.append(connected_entity_df)
            event_mentions_df2 = event_mentions_df2.drop_duplicates(subset=['Entity1Id', 'Entity2Id']).reset_index(
                drop=True)

            return event_mentions_df2

        # Append all excel files into a single table
        datalake_df = pd.DataFrame()
        dates_df = pd.DataFrame()
        for i in tqdm(range(len(file_list))):
            # Read the Input database
            transaction_df = pd.read_excel(input_path + file_list[i])[['JUMLAH URUSNIAGA', 'SENDER', 'RECEIVER']]
            datalake_date = pd.read_excel(input_path + file_list[i])[['TARIKH MASUK']]
            datalake_df = datalake_df.append(transaction_df, ignore_index=True)
            dates_df = dates_df.append(datalake_date, ignore_index=True)

        # Clean column 'JUMLAH URUSNIAGA'
        datalake_df['JUMLAH URUSNIAGA'] = datalake_df['JUMLAH URUSNIAGA'].str.replace('+', '')
        datalake_df['JUMLAH URUSNIAGA'] = datalake_df['JUMLAH URUSNIAGA'].str.replace('-', '')
        datalake_df['JUMLAH URUSNIAGA'] = datalake_df['JUMLAH URUSNIAGA'].str.replace(',', '')
        datalake_df['JUMLAH URUSNIAGA'] = datalake_df['JUMLAH URUSNIAGA'].astype('float')
        datalake_df['JUMLAH URUSNIAGA'] = datalake_df['JUMLAH URUSNIAGA'].apply(lambda x: f"RM {x}")

        # Transpose Documents from Datalake into EagleyeDB
        column_name = ['DocId', 'DocName', 'DocLocation', 'Date', 'Text']
        document_df = pd.DataFrame(
            {column_name[0]: [1], column_name[1]: ['AUDIT DATABASES'], column_name[2]: ['AUDIT SYSTEM/DATABASES'],
             column_name[3]: ['1/1/2022  12:00:00 AM'], column_name[4]: [' ']})
        document_df['Date'] = document_df['Date'].astype('datetime64')

        # Transpose Entity Type from Datalake into EagleyeDB
        column_name = ['TypeId', 'TypeName', 'TypeDescription']
        entity_type = get_entity_type(datalake_df)
        entity_type_desc = get_entity_desc(entity_type)
        entity_type_id = get_entity_type_id(datalake_df)
        entity_type_df2 = pd.DataFrame([entity_type_id, entity_type, entity_type_desc]).T
        entity_type_df2.columns = column_name
        # print(entity_type_df)

        # Transpose EntityAttribute from Datalake into Eagle Eye DB
        column_name = ['EntityId', 'AttributeName', 'AttributeValue', 'DocId']
        entity_attributes_df = pd.DataFrame(
            {column_name[0]: [''], column_name[1]: [''], column_name[2]: [''], column_name[3]: ['']})

        # Transpose Entities from Datalake into Eagle Eye DB
        column_name = ['EntityId', 'EntityName', 'EntityType', 'Location', 'Address', 'Image', 'DocId']
        entities = get_entity(datalake_df)
        entity_id = get_entity_id(entities)
        factor = get_factor2(entity_id)
        entity_type2 = get_entity_type2(entity_id, entity_type_id)
        location = get_location(factor)
        address = get_address(factor)
        image = get_image(entity_type2)
        doc_id = get_doc_id(factor)
        searcharound = get_search_around(entity_type2)
        searcharoundurl = get_search_around_url(factor)
        entity_df2 = pd.DataFrame([entity_id, entities, entity_type2, location, address, image, doc_id]).T
        entity_df2.columns = column_name
        entity_df2['EntityName'] = entity_df2['EntityName'].astype('string')
        entity_df2 = entity_df2.fillna('')
        entity_df2['EntityName'] = entity_df2['EntityName'].str.replace(' 00:00:00', '')
        # print(entity_df)

        # Transpose Entity Mentions from Datalake into Eagle Eye DB
        column_name = ['DocId', 'EntityId', 'StartIndex', 'EndIndex', 'Date', 'EntityUniqueId']
        doc_id = get_doc_id2(entity_id)
        entity_ids = get_entity_id2(entity_id)
        start_idx = get_start_idx(entity_id)
        end_idx = get_end_idx(entity_id)
        date = get_date(dates_df)
        entity_unique_id = get_entity_unique_id(doc_id, entity_ids, start_idx, end_idx)
        entity_mentions_df2 = pd.DataFrame([doc_id, entity_ids, start_idx, end_idx, date, entity_unique_id]).T
        entity_mentions_df2.columns = column_name
        entity_mentions_df2['Date'] = entity_mentions_df2['Date'].astype('datetime64')
        # print(entity_mentions_df)

        # Transpose Event Types from Datalake into Eagle Eye DB
        column_name = ['EventId', 'EventName', 'EventDescription']
        event_name = get_event_name()
        event_desc = get_event_desc(event_name)
        event_id = get_event_id(event_name)
        event_types_df2 = pd.DataFrame([event_id, event_name, event_desc]).T
        event_types_df2.columns = column_name
        # print(event_types_df2)

        # Transpose Event Mentions from Datalake into Eagle Eye DB
        entity1id, entity2id, entity_type1d, entity_type2d, merged_df = get_relations(entity_df2, entity_mentions_df2)
        doc_id = get_doc_id2(entity1id)
        date = get_date2(dates_df)
        event_mentions_df2 = pd.DataFrame([doc_id, entity1id, entity2id, entity_type1d, entity_type2d, date]).T
        column_name = get_columns()
        event_mentions_df2.columns = column_name
        event_mentions_df2['Date'] = event_mentions_df2['Date'].astype('datetime64')
        event_id = get_event_id2(event_mentions_df2)
        event_mentions_df2['EventId'] = event_id
        column_name = get_columns2()
        event_mentions_df2 = event_mentions_df2[column_name]
        # event_mentions_df2

        # Auto-connect among entities with attributes
        # event_mentions_df2, connected_entity_df = clean_na(event_mentions_df2, merged_df)
        # event_mentions_df2 = get_relations3(connected_entity_df, event_mentions_df2)
        # print(event_mentions_df2)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(output_path + 'EagleyeSampleDb.xlsx', engine='openpyxl')

        # Write each dataframe to a different worksheet.
        document_df.to_excel(writer, sheet_name='Documents', index=False)
        entity_type_df2.to_excel(writer, sheet_name='EntityTypes', index=False)
        entity_df2.to_excel(writer, sheet_name='Entities', index=False)
        entity_attributes_df.to_excel(writer, sheet_name='EntityAttributes', index=False)
        entity_mentions_df2.to_excel(writer, sheet_name='EntityMentions', index=False)
        event_types_df2.to_excel(writer, sheet_name='EventTypes', index=False)
        event_mentions_df2.to_excel(writer, sheet_name='EventMentions', index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

        # --------------------------------------------------------------------------------------------------------------
        # End of Python scripts
        # --------------------------------------------------------------------------------------------------------------

        obj = Post.objects.latest('id')
        with open('file-pdf-solid-240.png', 'rb') as destination_file:
            obj.processed_image.save('file-pdf-solid-240.png', File(destination_file), save=False)
        obj.save()
            
        obj = Post.objects.latest('id')
        context = {"processed_image": obj.processed_image}
        return context
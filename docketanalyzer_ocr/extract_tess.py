import io
import pandas as pd
import pdfplumber
from PIL import Image
import pytesseract
from pytesseract import Output


def process_page(page):
    image = page.to_image(resolution=300).original
    data_dict = pytesseract.image_to_data(
        image,
        config="-c preserve_interword_spaces=1x1 -c tessedit_do_invert=0 --psm 6 -l eng",
        output_type=Output.DICT,
    )
    data = pd.DataFrame(data_dict)
    data = data[(data['conf'] != -1)]
    data = data.sort_values(['block_num', 'par_num', 'line_num'])

    data['block'] = (
        data['block_num'].astype(str) + '__' +
        data['par_num'].astype(str)
    )
    data['line'] = (
        data['block'].astype(str) + '__' +
        data['line_num'].astype(str)
    )

    data = data.rename(columns={
        'left': 'x1', 'top': 'y1', 
        'text': 'content', 'conf': 'score',
    })
    print(data[['block', 'line', 'x1', 'y1', 'width', 'height']])

    scale_factor = 72.0 / 300.0  # Convert from 300 DPI to 72 DPI (PDF coordinates)
    data['x1'] = data['x1'] * scale_factor
    data['y1'] = data['y1'] * scale_factor
    data['width'] = data['width'] * scale_factor
    data['height'] = data['height'] * scale_factor

    print(data[['block', 'line', 'x1', 'y1', 'width', 'height']])

    #input()
    data['x2'] = data['x1'] + data['width']
    data['y2'] = data['y1'] + data['height']
    data = data[['line', 'block', 'x1', 'y1', 'x2', 'y2', 'content', 'score']]

    lines = data.groupby('line').apply(lambda x: pd.Series({
        'block': x['block'].iloc[0],
        'x1': x['x1'].min(),
        'y1': x['y1'].min(),
        'x2': x['x2'].max(),
        'y2': x['y2'].max(),
        'content': ' '.join(x['content'].astype(str)),
        'score': x['score'].mean()
    }))

    lines['bbox'] = lines.apply(lambda row:
        [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])]
    , axis=1)

    lines['spans'] = lines.apply(lambda row: [{
        'content': row['content'], 
        'bbox': row['bbox'],
        'score': row['score'],
        'type': 'text',
    }], axis=1)

    blocks = lines.groupby('block').apply(lambda x: pd.Series({
        'bbox': [
            int(x['x1'].min()), int(x['y1'].min()), 
            int(x['x2'].max()), int(x['y2'].max()),
        ],
        'lines': x[['bbox', 'spans']].to_dict('records'),
        'type': 'text',
    })).to_dict('records')

    for img in page.images:
        bbox = (
            img.get('x0', 0), img.get('y0', 0),
            img.get('x1', 0), img.get('y1', 0)
        )
        if all(v > 0 for v in bbox):
            blocks.append({
                'bbox': bbox,
                'type': 'image',
            })

    return blocks


def process_page_handled(page):
    try:
        return process_page(page)
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
    except RecursionError as e:
        print(f"RecursionError: {e}")
    except Image.DecompressionBombError as e:
        print(f"DecompressionBombError: {e}")
    return []


def process_with_tess(bits):
    try:
        with pdfplumber.open(io.BytesIO(bits)) as pdf:
            return [
                {'i': i, 'blocks': process_page_handled(page)} 
                for i, page in enumerate(pdf.pages)
            ]
    except Exception as e:
        print(f"Exception: {e}")
    return None

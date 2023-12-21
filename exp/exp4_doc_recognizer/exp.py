# import libraries
import os
import json
 import pandas as pd
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv()

endpoint = os.environ["AZURE_DI_ENDPOINT"] 
key = os.environ["AZURE_DI_KEY"] 

def format_bounding_region(bounding_regions):
    if not bounding_regions:
        return "N/A"
    return ", ".join("Page #{}: {}".format(region.page_number, format_polygon(region.polygon)) for region in bounding_regions)

def format_polygon(polygon):
    if not polygon:
        return "N/A"
    return ", ".join(["[{}, {}]".format(p.x, p.y) for p in polygon])


def analyze_general_documents():
    # sample document
    # docUrl = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf"
    docPath = "/home/sebacastillo/genai0/exp/exp4_doc_recognizer/Tesla_10k_short.pdf"

    # create your `DocumentAnalysisClient` instance and `AzureKeyCredential` variable
    document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    # poller = document_analysis_client.begin_analyze_document_from_url(
    #         "prebuilt-document", docUrl)    
    # https://learn.microsoft.com/en-us/python/api/azure-ai-formrecognizer/azure.ai.formrecognizer.documentanalysisclient?view=azure-python&viewFallbackFrom=azure-python-preview#azure-ai-formrecognizer-documentanalysisclient-begin-analyze-document
    with open(docPath, "rb") as f:
       poller = document_analysis_client.begin_analyze_document(
           "prebuilt-invoice", document=f, locale="en-US"
    )  

    result = poller.result()

    # save result https://learn.microsoft.com/en-us/python/api/azure-ai-formrecognizer/azure.ai.formrecognizer.analyzeresult?view=azure-python
    # result_dict = result.to_dict()
    # import json
    # with open("docrecognized.json", "w") as outfile: 
    #     json.dump(result_dict, outfile)

    for style in result.styles:
        if style.is_handwritten:
            print("Document contains handwritten content: ")
            print(",".join([result.content[span.offset:span.offset + span.length] for span in style.spans]))

    print("----Key-value pairs found in document----")
    for kv_pair in result.key_value_pairs:
        if kv_pair.key:
            print(
                    "Key '{}' found within '{}' bounding regions".format(
                        kv_pair.key.content,
                        format_bounding_region(kv_pair.key.bounding_regions),
                    )
                )
        if kv_pair.value:
            print(
                    "Value '{}' found within '{}' bounding regions\n".format(
                        kv_pair.value.content,
                        format_bounding_region(kv_pair.value.bounding_regions),
                    )
                )

    for page in result.pages:
        print("----Analyzing document from page #{}----".format(page.page_number))
        print(
            "Page has width: {} and height: {}, measured with unit: {}".format(
                page.width, page.height, page.unit
            )
        )

        for line_idx, line in enumerate(page.lines):
            print(
                "...Line # {} has text content '{}' within bounding box '{}'".format(
                    line_idx,
                    line.content,
                    format_polygon(line.polygon),
                )
            )

        for word in page.words:
            print(
                "...Word '{}' has a confidence of {}".format(
                    word.content, word.confidence
                )
            )

        for selection_mark in page.selection_marks:
            print(
                "...Selection mark is '{}' within bounding box '{}' and has a confidence of {}".format(
                    selection_mark.state,
                    format_polygon(selection_mark.polygon),
                    selection_mark.confidence,
                )
            )

    for table_idx, table in enumerate(result.tables):
        print(
            "Table # {} has {} rows and {} columns".format(
                table_idx, table.row_count, table.column_count
            )
        )
        for region in table.bounding_regions:
            print(
                "Table # {} location on page: {} is {}".format(
                    table_idx,
                    region.page_number,
                    format_polygon(region.polygon),
                )
            )
        for cell in table.cells:
            print(
                "...Cell[{}][{}] has content '{}'".format(
                    cell.row_index,
                    cell.column_index,
                    cell.content,
                )
            )
            for region in cell.bounding_regions:
                print(
                    "...content on page {} is within bounding box '{}'\n".format(
                        region.page_number,
                        format_polygon(region.polygon),
                    )
                )
    print("----------------------------------------")


if __name__ == "__main__":

    results = analyze_general_documents()
    
    results_dict = results.to_dict()
    with open("docrecognized.json", "w") as outfile: 
        json.dump(result_dict, outfile)

    print(results)

    # Explore tables        
    result = results
    for i, table in enumerate(result.tables):
       print(f"\nTable {i + 1} can be found on page:")
       for region in table.bounding_regions:
           print(f"...{region.page_number}")
       for cell in table.cells:
           print(
               f"Cell[{cell.row_index}][{cell.column_index}]: '{cell.content}'"
           )
   
    dataframes = []  # This will store the DataFrames for each table
    for table in result.tables:
        # Initialize a matrix (list of lists) to hold the cell data
        rows = [[] for _ in range(table.row_count)]

        for cell in table.cells:
            # Append the cell's text to the appropriate row
            rows[cell.row_index].append(cell.content)

        # Convert the matrix into a DataFrame
        table_df = pd.DataFrame(rows)
        dataframes.append(table_df)
    
    # Ensure the folder exists, create it if it does not
    os.makedirs("exp/exp4_doc_recognizer", exist_ok=True)
    # Loop through each DataFrame and save it as a CSV file
    for i, df in enumerate(dataframes):
        # Create a unique filename for each table
        filename = f"table_{i + 1}.csv"
        file_path = os.path.join("exp/exp4_doc_recognizer", filename)

        # Save the DataFrame as a CSV file
        df.to_csv(file_path, index=False)

    



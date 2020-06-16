python -m allennlp.service.server_simple \
    --archive-path C:\\Users\\jesse\\Documents\\Datago\\tmp\\test2\\model.tar.gz \
    --predictor text_classifier \
    --include-package AllenFrame \
    --title "文本分类" \
    --field-name sentence
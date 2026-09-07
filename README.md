Créer une selection de pages à partir d'un JSON output de Chandra : 

```bash
jq 'map(select(.page_index >= 114 and .page_index <= 326))' ./annuaires/bpt6k62915570/bpt6k62915570-5_835-ocr.json  > ./annuaires/bpt6k62915570/bpt6k62915570-114_327-ocr.json
```
# Charge le fichier json Almanach_Duverneuil_bpt6k124557p_1807-122_292_extracted.json
# Almanach_Duverneuil_bpt6k62915570_1804-119_331_extracted
import pydantic
from dataclasses import dataclass


class ExtractedData(pydantic.BaseModel):
    index: int
    label: str
    content: str | None
    native_label: str
    extracted_data: list[dict] | None
    bbox_2d: list[int]
    polygon: list[list[int]]


@dataclass
class Entry:
    order: int
    source_id: int
    description: str
    address: str
    category: str | None

    def __post_init__(self):
        # Trim, remove # and lowercase the category
        if self.category:
            self.category = self.category.replace("#", "").lower().strip()


if __name__ == "__main__":
    import json

    data = []
    with open("Almanach_Duverneuil_bpt6k124554j_1803-114_343_extraction.json", "r") as f:
        for d in json.load(f):
            data.extend(d)

    order = 1
    current_title = None
    entries = []
    for item in data:
        print(item)
        extracted_data = ExtractedData(**item)
        if "title" in extracted_data.native_label.lower():
            current_title = extracted_data.content
        else:
            for ed in extracted_data.extracted_data or []:
                entry = Entry(
                    order=order,
                    source_id=extracted_data.index,
                    description=ed["desc"],
                    address=ed["adresse"],
                    category=current_title,
                )
                order += 1
                entries.append(entry)
    print(entries)

    # Export to CSV
    import csv

    with open("Almanach_Duverneuil_bpt6k124554j_1803-114_343_extracted.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order", "source_id", "description", "address", "category"])
        for entry in entries:
            writer.writerow(
                [
                    entry.order,
                    entry.source_id,
                    entry.description,
                    entry.address,
                    entry.category,
                ]
            )

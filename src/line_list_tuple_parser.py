from langchain_core.output_parsers import BaseOutputParser


class LineListTupleOutputParser(BaseOutputParser[list[tuple[str, str]]]):
    def parse(self, text: str) -> list[tuple[str, str]]:
        lines = text.strip().split("\n")
        lines = list(filter(None, lines))
        separated_lines = [line.split(",", 1) for line in lines]
        return [tuple(map(str.strip, line_items)) for line_items in separated_lines]

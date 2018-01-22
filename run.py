from nps_extractor import NPSExtractor
from nus_extractor import NUSExtractor

extractors = []
#extractors.append(NPSExtractor())
extractors.append(NUSExtractor())

for extractor in extractors:
    extractor.extract()
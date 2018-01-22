from nps_extractor import NPSExtractor
from nus_extractor import NUSExtractor
from twitter_extractor import TwitterExtractor

extractors = []
#extractors.append(NPSExtractor())
extractors.append(NUSExtractor())
extractors.append(TwitterExtractor())

for extractor in extractors:
    extractor.extract()
"""
    Data Analysis and Query languages
    project - Task 2 : BaseLine Recommender System
    Mohammad A'arabi, Youssef El Hassani
    Summer Semester 2018
"""
import math
import json
import html
import sys
import numpy as np
try:
    from types import SimpleNamespace as Namespace
except ImportError:
    # Python 2.x fallback
    from argparse import Namespace

DEFAULT_SIM = 0.8


class BaselineRecommenderSystem:
    """A basic item-to-item Collaborative Filtering Recommender System
    """

    def __init__(self):
        """
        Creates an empty inverted index.
        """
        self.items = set()  # set of Items
        self.ratingsIndex = {}  # The ratings lists.
        self.itemsProperty = {}  # The item property lists.

    def loadRatings(self, ratingsFile):
        """This method allows to feed the RS with input ratings

        >>> engine = BaselineRecommenderSystem()
        >>> engine.loadRatings("example.csv")
        >>> index = sorted(engine.ratingsIndex.items())
        >>> [(item, [(i, '%.1f' % rating) for i, rating in l]) for item, l in index]
        ... #doctest: +NORMALIZE_WHITESPACE
        [('Alice', [('item1', '5.0'), ('item2', '3.0'), ('item3', '4.0'),
                    ('item4', '4.0'), ('item5', '1.0')]),
        ('User1', [('item1', '3.0'), ('item2', '1.0'), ('item3', '2.0'),
                   ('item4', '3.0'), ('item5', '1.0'), ('item6', '3.0')]),
        ('User2', [('item1', '4.0'), ('item2', '3.0'), ('item3', '4.0'),
                   ('item4', '3.0'), ('item5', '1.0'), ('item6', '5.0')]),
        ('User3', [('item1', '3.0'), ('item2', '3.0'), ('item3', '1.0'),
                   ('item4', '5.0'), ('item5', '1.0'), ('item6', '4.0')]),
        ('User4', [('item1', '1.0'), ('item2', '5.0'), ('item3', '5.0'),
                   ('item4', '2.0'), ('item5', '1.0'), ('item6', '1.0')])]
        """
        with open(ratingsFile, "r", encoding="utf-8") as f:
            for line in f:
                word = line.split(",")
                userId = word[0]
                itemId = word[1]
                # add item to the distinct set of items
                if itemId not in self.items:
                    self.items.add(itemId)
                rating = float(word[2])
                if userId not in self.ratingsIndex:
                    # The user is seen for first time, create new list.
                    self.ratingsIndex[userId] = [(itemId, rating)]
                else:
                    # The user already exists, append item to list
                    self.ratingsIndex[userId].append((itemId, rating))

    def loadItemsProperty(self, itemsPropertyFile):
        """This method allows to feed the RS with items properties

            For the sake of this unit test, I chose to test if the title was present.
            the object contains all the attributes, only the title was shown for the
            sake of visualisation.

         >>> engine = BaselineRecommenderSystem()
         >>> engine.loadItemsProperty("meta_example.json")
         >>> items = sorted(engine.itemsProperty.items())
         >>> [(item, html.unescape(object.title)) for item, object in items]
         ... #doctest: +NORMALIZE_WHITESPACE
         [('item1', 'Barnes & Noble HDTV Adapter Kit for NOOK HD and NOOK HD+'),
         ('item2', 'Barnes & Noble OV/HB-ADP Universal Power Kit'),
         ('item3', 'Audiovox Surface SURF402 Wet/Dry Screen Wipes'),
         ('item4', 'VideoSecu 24" Long Arm TV Wall Mount Low Profile Full Motion Cantilever Swing & Tilt wall bracket for most 22" to 55" LED LCD TV Monitor Flat Panel Screen VESA 200x200 400x400 up to 600x400mm - Articulating Arm Extend up to 24" MAH'),
         ('item5', 'Barnes & Noble Nook eReader - no 3G'),
         ('item6', 'Barnes & Noble Nook Simple Touch eBook Reader (Wi-Fi Only)')]
        """
        # Parse Json file
        with open(itemsPropertyFile, 'r', encoding="utf-8") as f:
            for line in f:
                # For each line, create a dict object out of the JSON line
                item = json.loads(line, object_hook=lambda d: Namespace(**d))
                # Check if item exists in the items property list
                if item.asin not in self.itemsProperty:
                    # assign the asin as the key and the object item as value
                    self.itemsProperty[item.asin] = item

    def predictRating(self, userId, itemId, similarity_threshold=DEFAULT_SIM, subset_fuzziness=0.8):
        """This method returns for a given user (the active user) and item a
        predicted rating.

        >>> engine = BaselineRecommenderSystem()
        >>> engine.loadRatings("example.csv")
        >>> engine.predictRating("Alice","item6")
        4.672172678162013
        """
        # Get the list of items already rated by userId
        relevantSet = set()
        ratingsSum = 0
        ratingsCount = 0
        for item, rating in self.ratingsIndex[userId]:
            relevantSet.add(item)
            ratingsSum += rating
            ratingsCount += 1
        relevantSet.add(itemId)
        ratingsMean = float(ratingsSum) / float(ratingsCount)
        # Find users that rated the same items
        ratersList = set()
        for user, item in self.ratingsIndex.items():
            potentialSet = {item[0] for item in item}
            if len(relevantSet.intersection(potentialSet)) > int(subset_fuzziness * len(relevantSet)) and user != userId:
                # print(len(relevantSet.intersection(potentialSet)), int(subset_fuzziness * len(relevantSet)), user != userId)
                # print('Yay!')
                ratersList.add(user)
        # Compute pearson similarities ( pearson was recommended in the lecture )
        similarities = {}
        vector1 = [rating[1] for rating in self.ratingsIndex[userId]]
        for rater in ratersList:
            vector2 = [rating[1] for rating in self.ratingsIndex[rater]]
            similarities[rater] = self.pearsonSimilarity(vector1, vector2)
        # Neighbor selection
        neighbors = []
        for user, similarity in similarities.items():
            # Select only users with more than 80% similarity by default
            if similarity > similarity_threshold:
                neighbors.append(user)
        # Check if neighbors list is empty
        if not neighbors:
            return "No neighbors found, try to change similarity value"
        # Value prediction
        similaritySum = 0.0
        neighborsMean = {}
        itemIdRatings = {}
        for user in neighbors:
            similaritySum += similarities[user]
            neighborSum = 0
            neighborCount = 0
            for item, rating in self.ratingsIndex[user]:
                if item in relevantSet:
                    if item == itemId:
                        itemIdRatings[user] = rating
                    neighborSum += rating
                    neighborCount += 1
            if neighborCount != 0:
                neighborsMean[user] = float(neighborSum) / float(neighborCount)
            else:
                return "Error: Cannot Divide by 0"
        # Calculating K
        k = 1/similaritySum
        predSum = 0
        # Calculating SubSum
        for user in neighbors:
            similarity = similarities[user]
            mean = neighborsMean[user]
            if user in itemIdRatings:
                rating = itemIdRatings[user]
                predSum += similarity * (rating - mean)
        return ratingsMean + k * predSum

    def predictTopKRecommendations(self, userId, k, similarity_threshold=DEFAULT_SIM, subset_fuzziness=0.8):
        """Returns for the active user a list of k recommendations.

        >>> engine = BaselineRecommenderSystem()
        >>> engine.loadRatings("example.csv")
        >>> engine.predictTopKRecommendations("Alice",1)
        [('item6', 4.672172678162013)]
        """
        # Get the items that were not rated by
        ratedItems = [items[0] for items in self.ratingsIndex[userId]]
        unratedItems = list(self.items - set(ratedItems))
        if not unratedItems:
            return "No unrated items found"
        # Predict ratings for unrated items
        result = []
        for item in unratedItems:
            rating = self.predictRating(userId, item, similarity_threshold, subset_fuzziness)
            result.append((item, rating))
        # Sort list based on ratings (descending order)
        try:
            result.sort(key=lambda tup: tup[1], reverse=True)
        except Exception:
            pass
        # Trim result list if it's larger than k
        if len(result) > k:
            return result[:k]
        # else if it's less or equal to k then return the list as it is
        return result

    def cosineSimilarity(self, vector1, vector2):
        """compute cosine similarity of two vectors X and Y
        given the following formula: (X dot Y)/{||X||*||Y||)

        >>> vector1 = [5, 3, 4, 4, 1]
        >>> vector2 = [3, 1, 2, 3, 1]
        >>> engine = BaselineRecommenderSystem()
        >>> engine.cosineSimilarity(vector1,vector2)
        0.972571602704942
        """
        xx, xy, yy = 0, 0, 0
        for i in range(len(vector1)):
            x = vector1[i]
            y = vector2[i]
            xy += x * y
            xx += x * x
            yy += y * y
        return xy/math.sqrt(xx * yy)

    def pearsonSimilarity(self, vector1, vector2):
        """compute cosine similarity of two vectors X and Y
        given the following formula: (X dot Y)/{||X||*||Y||)

        >>> vector1 = [5, 3, 4, 4, 1]
        >>> vector2 = [3, 1, 2, 3, 1]
        >>> engine = BaselineRecommenderSystem()
        >>> engine.pearsonSimilarity(vector1,vector2)
        0.8242255917447336
        """
        n = len(vector1)
        avg_x = float(sum(vector1)) / len(vector1)
        avg_y = float(sum(vector2)) / len(vector2)
        covariance = 0
        xdiff2 = 0
        ydiff2 = 0
        for i in range(n):
            try:
                xdiff = vector1[i] - avg_x
                ydiff = vector2[i] - avg_y
                covariance += xdiff * ydiff
                xdiff2 += xdiff * xdiff
                ydiff2 += ydiff * ydiff
            except IndexError:
                pass

        try:
            return covariance / math.sqrt(xdiff2 * ydiff2)
        except ZeroDivisionError:
            return np.infty

    def renderOutput(self, result):
        """
        Render the output for the given result. Fetch the
        the titles and descriptions, and prices of the recommended documents.

        >>> engine = BaselineRecommenderSystem()
        >>> engine.loadRatings("example.csv")
        >>> engine.loadItemsProperty("meta_example.json")
        >>> result = engine.predictTopKRecommendations("Alice",1)
        >>> engine.renderOutput(result)
        ... #doctest: +NORMALIZE_WHITESPACE
        <BLANKLINE>
        [0mitem6[1
        [1mBarnes & Noble Nook Simple Touch eBook Reader (Wi-Fi Only)[0m
        Barnes   Noble Nook Simple Touch Wi Fi ReaderIncredibly easy just touch and read  Ultra light  thin and the longest battery life  Most advanced E Ink 6  display w  crisp fonts  Expert recommendations and fun social features Features   Easy to Use 6  Touchscreen  Shop for the best books  magazines and newspapers right on your NOOK with just a touch  Turn pages  look up words  highlight passages  adjust the font size and style just by tapping the infrared powered touchscreen   Clear  Crisp Reading  NOOK features the most advanced E Ink Pearl technology  The high contrast 16 level grayscale touchscreen displays text as crisp and clear as a printed page  so you can read easily even in bright sun  50  more contrast than NOOK 1st Edition   Ultra Light  Ultra Portable  At 7 48 ounces  NOOK islighter than a paperback and super thin  yet holds up to 1 000 books  magazines and newspapers so it s easy to take your entire library with you anywhere   Longest Battery Life  With the longest battery life of any eReader  you can read for up totwo months on just one charge  That s enough timeto start and finish a lot of great stories or an entire series   World s Largest Bookstore  Over2 million titles including books  magazines   newspapers  just a touch away  Thousands are FREE  most others  9 99 or less  Pre order books and subscribe to magazines so you ll get them the instant they re released   Read Your Way  Make the text bigger or smaller   See demo  Choose the font style you like  You can even add bookmarks and highlight passages while you read   Immersive Reading Experience  With 80  less flashing than other eReaders  NOOK delivers an immersive reading experience so it s easy to lose yourself in your latest read  And with Fast Page  just touch and hold the side button to quickly scan through any book  magazine or newspaper to get to where you want to read lightning fast   NOOK Friends  Connect with NOOK Friends to share and fi
        Price : 79.49
        <BLANKLINE>
        # total recommended items: 1.
        """

        # Iterate over results
        for item, rating in result:
            itemObject = self.itemsProperty[item]
            asin = item
            title = itemObject.title
            desc = itemObject.description
            price = itemObject.price
            # Replace html elements into string
            if title is not None:
                title = html.unescape(title)
            # Print the asin in red.
            asin = "\033[0m%s\033[1" % asin
            # Print the the title in bold.
            title = "\033[1m%s\033[0m" % title
            # reformat price before printing
            price = "Price : " + str(price)
            print("\n%s\n%s\n%s\n%s" % (asin, title, desc, price))

        print("\n# total recommended items: %s." % len(result))


if __name__ == "__main__":
    # Parse the command line arguments.
    if len(sys.argv) < 3:
        print(
            "Usage: python3 BaselineRecommenderSystem.py <rating file> <item metadata file>")
        sys.exit()
    fileRatings = sys.argv[1]
    fileMetaData = sys.argv[2]

    # Create a new dict out of the rating file
    try:
        print("Reading from file '%s'..." % fileRatings)
        engine = BaselineRecommenderSystem()
        engine.loadRatings(fileRatings)
        print("Ratings successfully loaded")
        print("Reading from file '%s'..." % fileMetaData)
        print("Reading Items MetaData file")
        engine.loadItemsProperty(fileMetaData)
        print("Item properties successfully loaded")

    except IOError:
        print("Error: Could not load files")
        quit()

    while True:
        # Ask for a user query.
        query = input("\nYour selected User: ")
        # Ask for the top k value
        k = input("\nNumber of desired recommendations: ")
        # Process the query.
        try:
            result = engine.predictTopKRecommendations(query, int(k))
            # Render the output.
            engine.renderOutput(result)
        except ValueError:
            print("Error: User Not Found")

"""
    Data Analysis and Query languages
    Project - Task 2 : Advanced Recommender System
    Mohammad A'arabi, Youssef El Hassani
    Summer Semester 2018
"""
import math
import json
import html
import sys
import numpy

from scipy.sparse import csr_matrix

try:
    from types import SimpleNamespace as Namespace
except ImportError:
    # Python 2.x fallback
    from argparse import Namespace

# Default lambda value
DEFAULT_LAMBDA = 0.1
# Default number of factors for X and Y
DEFAULT_FACTORS = 1

class AdvancedRecommenderSystem:
    """An advanced item-to-item Collaborative Filtering Recommender System
    Using matrix factorization
    """

    def __init__(self):
        """
        Creates an empty inverted index.
        """
        self.items = set()  # set of Items
        self.ratingsIndex = {}  # The ratings lists.
        self.itemsProperty = {}  # The item property lists.
        self.ur_matrix = None  # The user-rating matrix.
        self.user_indices_in_matrix = {}  # Row indices in the matrix per user.
        self.item_indices_in_matrix = {}  # Column indices in the matrix per term.
        self.usersCount = 0  # Number of user in the user item matrix.
        self.itemsCount = 0  # Number of items in the user item matrix.
        self.userVectors = None  # User Vectors.
        self.itemVectors = None  # Item Vectors.
        # Setting printing options for matrices
        numpy.set_printoptions(formatter={'float': lambda x: ("%.3f" % x)})

    def loadRatings(self, ratingsFile):
        """This method allows to feed the RS with input ratings
            The result will be a user x items matrix where missing values are initialized with zeros.
        
        >>> engine = AdvancedRecommenderSystem() # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> engine.loadRatings("example.csv")
        >>> print(numpy.round(engine.ur_matrix.todense().tolist(), 3))
        [[5.000 3.000 4.000 4.000 1.000 0.000]
         [3.000 1.000 2.000 3.000 1.000 3.000]
         [4.000 3.000 4.000 3.000 1.000 5.000]
         [3.000 3.000 1.000 5.000 1.000 4.000]
         [1.000 5.000 5.000 2.000 1.000 1.000]]
        """
        itemCount = 0
        # Fill the ratings Index from file.
        with open(ratingsFile, "r", encoding="utf-8") as f:
            for line in f:
                word = line.split(",")
                userId = word[0]
                itemId = word[1]
                # add item to the distinct set of items.
                if itemId not in self.items:
                    self.items.add(itemId)
                #add corresponding matrix index to a dict
                if itemId not in self.item_indices_in_matrix:
                    self.item_indices_in_matrix[itemId] = itemCount
                    itemCount += 1
                rating = float(word[2])
                if userId not in self.ratingsIndex:
                    # The user is seen for first time, create new list.
                    self.ratingsIndex[userId] = [(itemId, rating)]
                else:
                    # The user already exists, append item to list.
                    self.ratingsIndex[userId].append((itemId, rating))
        # Build user-rating matrix from the ratings index.
        rows = []
        cols = []
        vals = []
        for i, user in enumerate(self.ratingsIndex):
            self.user_indices_in_matrix[user] = i
            for item, rating in self.ratingsIndex[user]:
                rows.append(i)
                cols.append(self.item_indices_in_matrix[item])  # get column index of item.
                vals.append(rating)

        # Create the sparse user-rating matrix, use float values due to ratings.
        self.ur_matrix = csr_matrix((vals, (rows, cols)), dtype=float)
        self.usersCount, self.itemsCount = self.ur_matrix.shape

    def loadItemsProperty(self, itemsPropertyFile):
        """This method allows to feed the RS with items properties

            For the sake of this unit test, I chose to test if the title was present.
            the object contains all the attributes, only the title was shown for the
            sake of visualisation.

         >>> engine = AdvancedRecommenderSystem()
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
                # For each line, create a dict object out of the JSON line.
                item = json.loads(line, object_hook=lambda d: Namespace(**d))
                # Check if item exists in the items property list.
                if item.asin not in self.itemsProperty:
                    # assign the asin as the key and the object item as value.
                    self.itemsProperty[item.asin] = item
    
    def iterAls (self, k):
        """This method performs the alternating least squares algorithm over k iterations

            This method cannot be tested using a unit test because the numbers generated are random
        """
        self.userVectors = numpy.random.random((self.usersCount, DEFAULT_FACTORS))
        self.itemVectors = numpy.random.random((self.itemsCount, DEFAULT_FACTORS))

        while k > 0:
            self.userVectors = self.als(self.userVectors,self.itemVectors,DEFAULT_LAMBDA,"user")
            self.itemVectors = self.als(self.userVectors,self.itemVectors,DEFAULT_LAMBDA,"item")
            k -= 1

    def als (self, X, Y, lambda_,type_):
        """This method computes on step of the als algorithm. The purpose is to compute the latent vectors by user.
        This function performs one step based on users and returns the user vectors

        >>> engine = AdvancedRecommenderSystem() # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> engine.loadRatings("example.csv")
        >>> X = numpy.array([[0.452],[0.627],[0.851],[0.628],[0.652]])
        >>> Y = numpy. array([[0.881],[0.216],[0.325],[0.435],[0.302],[0.456]])
        >>> als_step = engine.als(X,Y,0.1,"item")
        >>> print(als_step)
        [[4.498]
         [4.319]
         [4.620]
         [4.768]
         [1.432]
         [4.150]]
        """
        if type_ == "user":
            # Compute Y transpose * Y
            YTY = Y.T.dot(Y)
            # Compute lambda * I (identity)
            lambdaIden = numpy.eye(YTY.shape[0]) * lambda_
            # Solve x(i)= (YTY+λI)−1 * YT * r(i) for all i in userVectors
            for user in range(X.shape[0]):
                X [user, :] = numpy.linalg.solve((YTY + lambdaIden), self.ur_matrix[user, :].dot(Y))
            return X
        elif type_ == "item":
            # Compute X transpose * X
            XTX = X.T.dot(X)
            # Compute lambda * I (identity)
            lambdaIden = numpy.eye(XTX.shape[0]) * lambda_
            # Solve y(i)= (XTX+λI)−1 * XT * r(i) for all i in userVectors
            for item in range(Y.shape[0]):
                Y [item, :] = numpy.linalg.solve((XTX + lambdaIden), self.ur_matrix[:, item].T.dot(X))
            return Y

    def predictRating(self, userId, itemId):
        """This method returns for a given user (the active user) and item a
        predicted rating.

        >>> engine = AdvancedRecommenderSystem() # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> engine.loadRatings("example.csv")
        >>> engine.iterAls(200)
        >>> engine.predictRating("Alice","item6")
        2.799403532482411
        """
        userIndex = self.user_indices_in_matrix[userId]
        itemIndex = self.item_indices_in_matrix[itemId]
        return self.userVectors[userIndex, :].dot(self.itemVectors[itemIndex, :].T)

    def predictTopKRecommendations(self, userId, k):
        """Returns for the active user a list of k recommendations.

        >>> engine = AdvancedRecommenderSystem()
        >>> engine.loadRatings("example.csv")
        >>> engine.predictTopKRecommendations("Alice",1)
        [('item6', 4.672172678162013)]
        
        """
        # Get the items that were not rated by the user
        ratedItems = [items[0] for items in self.ratingsIndex[userId]]
        unratedItems = list(self.items - set(ratedItems))
        if not unratedItems:
            return "No unrated items found"
        # Predict ratings for unrated items
        result = []
        for item in unratedItems:
            rating = self.predictRating(userId, item)
            result.append((item, rating))
        # Sort list based on ratings (descending order)
        result.sort(key=lambda tup: tup[1], reverse=True)
        # Trim result list if it's larger than k
        if len(result) > k:
            return result[:k]
        # else if it's less or equal to k then return the list as it is
        return result

    def renderOutput(self, result):
        """
        Render the output for the given result. Fetch the
        the titles and descriptions, and prices of the recommended documents.
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
            "Usage: python3 AdvancedRecommenderSystem.py <rating file> <item metadata file>")
        sys.exit()
    fileRatings = sys.argv[1]
    fileMetaData = sys.argv[2]

    # Create a new dict out of the rating file
    try:
        print("Reading from file '%s'..." % fileRatings)
        engine = AdvancedRecommenderSystem()
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

import sys
import math

from crossword import *

"""
Shuyan Liu
CS50's Intro to AI
07/07/2020
"""


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v in self.domains:
            newDomain = self.domains[v].copy()
            for word in self.domains[v]:
                if len(word) != v.length:
                    newDomain.remove(word)
            self.domains[v] = newDomain

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]
        newDomain = self.domains[x].copy()
        for wordX in self.domains[x]:
            noValue = True
            for wordY in self.domains[y]:
                if wordX[overlap[0]] == wordY[overlap[1]]:
                    noValue = False
            if noValue:
                newDomain.remove(wordX)
                revised = True
        self.domains[x] = newDomain
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs == None:
            arcs = []
            for i in self.domains:
                for j in self.domains:
                    if i != j and self.crossword.overlaps[i, j] != None:
                        arcs.append((i, j))
        while len(arcs) != 0:
            arc = arcs.pop(0)
            if self.revise(arc[0], arc[1]):
                if len(self.domains[arc[0]]) == 0:
                    return False
                for z in self.crossword.neighbors(arc[0]):
                    if z != arc[1]:
                        arcs.append((z, arc[0]))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for v in self.domains:
            if v not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Check for overlaps
        for v1, v2 in self.crossword.overlaps:
            if v1 in assignment and v2 in assignment:
                overlap = self.crossword.overlaps[v1, v2]
                if overlap != None and assignment[v1][overlap[0]] != assignment[v2][overlap[1]]:
                    return False
        # Check for word length and uniqueness
        usedWords = []
        for v in assignment:
            word = assignment[v]
            if len(word) != v.length or word in usedWords:
                return False
            else:
                usedWords.append(word)
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        values = []
        for v in self.domains[var]:
            values.append([v, 0]) # Add all values as a list containing the value and
                                  # the number of neighboring values they rule out
        for i in range(len(values)):
            # Find all conflicts
            conflict = 0
            for v in self.crossword.neighbors(var):
                if v not in assignment:
                    overlap = self.crossword.overlaps[var, v]
                    for word in self.domains[v]:
                        if values[i][0][overlap[0]] != word[overlap[1]]:
                            conflict += 1
            values[i][1] = conflict
        # Sort the values according the number of conflicts, in ascending order
        values = sorted(values, key=lambda value: value[1])
        for i in range(len(values)):
            values[i] = values[i][0] # Remove the number
        return values
            

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        variables = []
        # Check for minimum number of remaining values
        minimum = math.inf
        for v in self.domains:
            if v not in assignment:
                value = len(self.domains[v])
                if value == minimum:
                    variables.append(v)
                elif value < minimum:
                    variables.clear()
                    variables.append(v)
                    minimum =  value
        # If there is only one value, then return it
        if len(variables) == 1:
            return variables[0]
        # Otherwise check for the variable with the highest degree
        maximum = -math.inf
        for v in variables:
            degree = len(self.crossword.neighbors(v))
            if degree > maximum:
                bestChoice = v
                maximum = degree
        return bestChoice
                

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            assignment.update({var: value})
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result != None:
                    return result
            assignment.pop(var)
        return None
    

def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()

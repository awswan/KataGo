from gamestate import GameState
from board import Board
import random

SAMPLE_SIZE=8

def shapley_sample(population, valuation):
    """Shapley sampling algorithm

    See Castro, Gómez, Tejada, Polynomial calculation of the Shapley value based on sampling,
    https://doi.org/10.1016/j.cor.2008.04.004    
    """
    mask = set()
    baseline = valuation(mask)
    value_store = {key: 0.0 for key in population}

    for _ in range(SAMPLE_SIZE):
        random.shuffle(population)
        mask.clear()
        previous = baseline

        for p in population:
            mask.add(p)
            v = valuation(mask)
            value_store[p] += (v - previous) / SAMPLE_SIZE
            previous = v
            
        return value_store

def single_board_lead(model, board, mask, rules=GameState.RULES_TT):
    gs = GameState(19,rules)
    gs.board = board.copy_masked(mask)
    gs.boards = [board.copy_masked(mask)]
    outputs = gs.get_model_outputs(model)
    return outputs['lead']
    
def explain_lead(model, board, flip_opp=True):
    """Apply Shapley sampling to overall lead for stones on board.

    The lead of the current player is split between the stones present on the board.
    The share of each stone is 'fair' in the sense that stones contributing more to
    the overall lead get a larger share and those contributing less get a smaller share.
    The opponent's stones also contribute towards the overall lead. Typically, they will
    contribute a negative amount.
    """
    locs = board.get_stone_locs()
    
    res = shapley_sample(locs, lambda mask: single_board_lead(model, board, mask))
    if flip_opp:
        for loc in locs:
            if board.board[loc] != board.pla:
                res[loc] = - res[loc]

    return res

def explain_diff(model, board, loc1, loc2):
    """Apply Shapley sampling to the score difference between two positions.

    We take two locations loc1 and loc2 that are available to play and consider the
    score difference between adding one verses the other to the board. We divide this
    score difference amongst the stones already on the board, to give an explanation
    why one is better than the other in the current board position.
    """
    locs = board.get_stone_locs()
    board1 = board.copy()
    board1.add_unsafe(board.pla, loc1)
    board2 = board.copy()
    board2.add_unsafe(board.pla, loc2)

    def valuation(mask):
        mask.add(loc1)
        v1 = single_board_lead(model, board1, mask)
        mask.discard(loc1)
        mask.add(loc2)
        v2 = single_board_lead(model, board2, mask)
        mask.discard(loc2)
        return v1 - v2

    return shapley_sample(locs, valuation)

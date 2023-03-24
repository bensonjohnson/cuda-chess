def piece_to_index(piece):
    piece_type = piece.piece_type
    color = piece.color
    index = piece_type - 1
    if color == chess.BLACK:
        index += 6
    return index

def board_to_tensor(board):
    tensor = torch.zeros(12, 8, 8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            tensor[piece_to_index(piece), square // 8, square % 8] = 1
    return tensor

def preprocess_data(games):
    positions = []
    results = []

    for game in games:
        board = game.board()
        result = game.headers["Result"]

        if result == "1-0":
            result_value = 1
        elif result == "0-1":
            result_value = -1
        else:
            result_value = 0

        for move in game.mainline_moves():
            positions.append(board_to_tensor(board))
            results.append(result_value)
            board.push(move)

    positions = torch.stack(positions)
    results = torch.tensor(results, dtype=torch.long)

    return positions, results

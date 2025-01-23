import asyncio
import json
import websockets
from queue import PriorityQueue
import random
import os
import getpass

# Map dimensions
MAP_WIDTH = 48
MAP_HEIGHT = 24

# Define the four reference points
reference_points = [
    (3, 3),
    (3, MAP_HEIGHT - 4),
    (MAP_WIDTH - 4, 3),
    (MAP_WIDTH - 4, MAP_HEIGHT - 4)
]

# Global variables for target point and sweeping path
target_point = None
sweeping_path = []

snake_state = 'sweeping'  # 'sweeping', 'to_food', 'returning'
food_target = None
path_to_food = []

steps_since_last_superfood = 0

# Variable to store obstacles
obstacles = {}

# Variable to store the current direction
current_direction = None  # 'w', 'a', 's', or 'd'

can_traverse = True

def a_star_search(start, goal, mapa, obstaculos, can_traverse):
    """
    Algoritmo A* para encontrar o caminho mais curto entre dois pontos.
    """
    open_list = PriorityQueue()
    open_list.put((0, start))
    came_from = {}
    g_score = {start: 0}

    while not open_list.empty():
        current = open_list.get()[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, mapa, obstaculos, can_traverse):
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                open_list.put((f_score, neighbor))

    return None

def heuristic(pos, goal):
    """
    Função heurística para estimar a distância entre dois pontos.
    """
    if pos is None or goal is None:
        return float('inf') 
    dx = min(abs(pos[0] - goal[0]), MAP_WIDTH - abs(pos[0] - goal[0]))
    dy = min(abs(pos[1] - goal[1]), MAP_HEIGHT - abs(pos[1] - goal[1]))
    return dx + dy

def reconstruct_path(came_from, current):
    """
    Reconstrói o caminho do início até o nó atual.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def get_neighbors(state, mapa, obstaculos, can_traverse):
    """
    Retorna as posições vizinhas do estado atual.
    """
    x, y = state
    width, height = mapa
    neighbors = []
    potential_moves = []

    if can_traverse:
        potential_moves = [
            ((x + 1) % width, y),
            ((x - 1) % width, y),
            (x, (y + 1) % height),
            (x, (y - 1) % height),
        ]
    else:
        potential_moves = [
            (x + 1, y) if (x + 1) < width else None,
            (x - 1, y) if (x - 1) >= 0 else None,
            (x, y + 1) if (y + 1) < height else None,
            (x, y - 1) if (y - 1) >= 0 else None,
        ]
    
        potential_moves = [pos for pos in potential_moves if pos is not None]

    for new_pos in potential_moves:
        if new_pos in obstaculos:
            if obstaculos[new_pos] == "wall" and not can_traverse:
                continue
            elif obstaculos[new_pos] == "body":
                continue
        neighbors.append(new_pos)
    return neighbors

async def agent_loop(server_address="localhost:8001", agent_name="student"):
    """Loop principal do agente AI."""
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))

        while True:
            try:
                state_msg = await websocket.recv()
                state = json.loads(state_msg)

                if 'body' in state:
                    next_move = calculate_next_move(state)
                    if next_move:
                        await websocket.send(json.dumps({"cmd": "key", "key": next_move}))
                elif 'highscores' in state:
                    print("Recebido highscores:", state['highscores'])
                else:
                    print("Estado recebido não contém 'body'. Aguardando o próximo estado...")

            except websockets.exceptions.ConnectionClosedOK:
                print("Servidor terminou a conexão.")
                return

def calculate_next_move(state):
    """
    Calcula o próximo movimento com base no estado atual do jogo.
    """
    global target_point, sweeping_path, obstacles, current_direction, can_traverse, steps_since_last_superfood
    global snake_state, food_target, path_to_food

    steps_since_last_superfood += 1

    snake_body = state['body']  
    head = tuple(snake_body[0])  
    body = [tuple(b) for b in snake_body[1:]]  
    sight = state['sight']
    vision_range = state['range']

    can_traverse = state.get('traverse', True)
    #print(f"Atualizado can_traverse: {can_traverse}")
    update_obstacles(sight, body)

    predicted_positions = predict_other_snake_moves(sight)
    temporary_obstacles = obstacles.copy()
    for pos in predicted_positions:
        temporary_obstacles[pos] = "predicted_body"

    while True:
        if snake_state == 'sweeping':
            if target_point is None:
                target_point = get_new_target_point(head)

                path_to_target = a_star_search(head, target_point, (MAP_WIDTH, MAP_HEIGHT), temporary_obstacles, can_traverse)
                if path_to_target and len(path_to_target) > 1:
                    sweeping_path.extend(path_to_target[1:]) 
                else:
                    target_point = get_new_target_point(head, exclude=target_point)
                    return calculate_next_move(state)
                
            food_positions = []
            for x_str, col in sight.items():
                for y_str, tile in col.items():
                    x, y = int(x_str), int(y_str)

                    if tile == 2:  # Comida normal
                        food_positions.append((x, y))

                    if vision_range == 6 and can_traverse:  # Super foods quando vision_range é 6 e pode atravessar
                        if tile == 3:
                            temporary_obstacles[(x, y)] = "predicted_body"

                    if not can_traverse:  # Super foods para retomar traverse true
                        if tile == 3:
                            food_positions.append((x, y))
                            steps_since_last_superfood = 0

                    if vision_range == 2:      # Super foods para retomar visão normal ou superior
                        if tile == 3:
                            food_positions.append((x, y))
                            steps_since_last_superfood = 0

                    if steps_since_last_superfood >= 300:       # Se passaram 300 passos desde a última superfood, considera comer superfoods
                        if tile == 3:
                            food_positions.append((x, y))
                            steps_since_last_superfood = 0

            if food_positions:
                distances = [heuristic(head, food) for food in food_positions]
                min_distance = min(distances)
                food_target = food_positions[distances.index(min_distance)]
                #print(f"Comida mais próxima encontrada: {food_target}")

                path_to_food = a_star_search(head, food_target, (MAP_WIDTH, MAP_HEIGHT), temporary_obstacles, can_traverse)
                if path_to_food and len(path_to_food) > 1:
                    path_to_food = path_to_food[1:]  
                    snake_state = 'to_food'  
                    #print(f"Estado alterado para 'to_food'. Caminho para a comida: {path_to_food}")
                    return calculate_next_move(state)
                else:
                    #print("Não foi possível encontrar caminho para a comida. Continuar sweeping.")
                    continue
            else:
                #print("Nenhuma comida à vista. Continuar sweeping.")
                return continue_sweeping(state, head, body)

        elif snake_state == 'to_food':
            if path_to_food:
                next_step = path_to_food.pop(0)
                if is_move_safe(next_step, body):
                    proposed_direction = direction_from_steps(head, next_step, can_traverse)
                    if not is_opposite_direction(current_direction, proposed_direction):
                        current_direction = proposed_direction
                        return proposed_direction
                    else:
                        next_move = choose_new_direction(snake_body, body)
                        if not next_move:
                            #print("Nenhuma direção viável. Recalculando estratégia...")
                            sweeping_path.clear()
                            target_point = None
                            snake_state = 'sweeping'
                            return calculate_next_move(state)
                        return next_move
                else:
                    path_to_food = a_star_search(head, food_target, (MAP_WIDTH, MAP_HEIGHT), temporary_obstacles, can_traverse)
                    if path_to_food and len(path_to_food) > 1:
                        path_to_food = path_to_food[1:]
                        return calculate_next_move(state)
                    else:
                        snake_state = 'returning'
                        #print("Não foi possível encontrar caminho para a comida. Retornando ao estado 'returning'.")
                        return calculate_next_move(state)
            else:
                snake_state = 'returning'
                return calculate_next_move(state)

        elif snake_state == 'returning':
            if sweeping_path:
                index, closest_point = find_closest_point_ahead(head, sweeping_path)
                if closest_point:
                    path_back = a_star_search(head, closest_point, (MAP_WIDTH, MAP_HEIGHT), temporary_obstacles, can_traverse)
                    if path_back and len(path_back) > 1:
                        path_back = path_back[1:]  
                        sweeping_path = path_back + sweeping_path[index+1:]
                        snake_state = 'sweeping'  
                        return continue_sweeping(state, head, body)
                    else:
                        sweeping_path.clear()
                        target_point = None
                        snake_state = 'sweeping'
                        continue
                else:
                    sweeping_path.clear()
                    target_point = None
                    snake_state = 'sweeping'
                    continue
            else:
                sweeping_path.clear()
                target_point = None
                snake_state = 'sweeping'
                continue

        else:
            sweeping_path.clear()
            target_point = None
            snake_state = 'sweeping'
            continue

def continue_sweeping(state, head, body):
    """
    Continua o caminho de sweeping para explorar o mapa.
    """
    global sweeping_path, current_direction, target_point, can_traverse

    while True:
        if sweeping_path:
            next_target = sweeping_path[0]
            temporary_obstacles = obstacles.copy()
            for b in body:
                temporary_obstacles[b] = "body"
            path = a_star_search(head, next_target, (MAP_WIDTH, MAP_HEIGHT), temporary_obstacles, can_traverse)
            if path and len(path) > 1:
                next_step = path[1]
                proposed_direction = direction_from_steps(head, next_step, can_traverse)
                if is_move_safe(next_step, body):
                    if not is_opposite_direction(current_direction, proposed_direction):
                        current_direction = proposed_direction
                        if next_step == next_target:
                            sweeping_path.pop(0)
                        return proposed_direction
            sweeping_path.pop(0)
        else:
            if target_point:
                sweeping_path = generate_sweeping_path(state, target_point, current_direction, body)
                if sweeping_path:
                    continue

            return choose_new_direction([head], body)

def find_closest_point_ahead(head, path):
    """
    Encontra o ponto mais próximo no caminho à frente da posição atual da cobra.
    """
    min_distance = float('inf')
    closest_point = None
    index = None
    for i, point in enumerate(path):
        distance = heuristic(head, point)
        if distance < min_distance:
            direction_to_point = direction_from_steps(head, point, can_traverse)
            if direction_to_point != opposite_direction(current_direction):
                min_distance = distance
                closest_point = point
                index = i
    return index, closest_point

def opposite_direction(direction):
    """
    Retorna a direção oposta da direção especificada.
    """
    if direction is None:
        return None
    opposite = {
        "w": "s",
        "s": "w",
        "a": "d",
        "d": "a"
    }
    return opposite.get(direction)

def get_new_target_point(head, exclude=None):
    """
    Retorna um novo ponto alvo para o caminho de sweeping.
    """
    distances = []
    points = []
    for rp in reference_points:
        if rp != exclude:
            distances.append(heuristic(head, rp))
            points.append(rp)
    if not points:
        return None 
    min_distance = min(distances)
    target = points[distances.index(min_distance)]
    return target

def generate_sweeping_path(state, start_point, current_direction, body):
    """
    Gera um caminho de sweeping para explorar o mapa.
    """
    vision_range = state['range']
    path = []
    body_positions = set(body)
    left = vision_range
    right = MAP_WIDTH - vision_range  
    top = vision_range - 1
    bottom = MAP_HEIGHT - vision_range

    x_positions = []
    x = left
    while x <= right:
        x_positions.append(x)
        x += vision_range * 2  

    if x_positions[-1] != right:
        x_positions.append(right)

    if current_direction == "a" or (current_direction is None and start_point[0] > MAP_WIDTH // 2):
        x_positions.reverse()

    for i, x in enumerate(x_positions):
        if current_direction == "w" or (current_direction is None and start_point[1] > MAP_HEIGHT // 2):
            y_range = list(range(bottom, top -1, -1)) if i % 2 == 0 else list(range(top, bottom + 1))
        else:
            y_range = list(range(top, bottom + 1)) if i % 2 == 0 else list(range(bottom, top -1, -1))

        for y in y_range:
            pos = (x, y)
            if (pos not in obstacles or obstacles[pos] != "wall") and (pos not in body_positions):
                path.append(pos)

        if i < len(x_positions) -1:
            next_x = x_positions[i+1]
            y = path[-1][1] if path else start_point[1]
            step = 1 if next_x > x else -1
            xh_range = list(range(x + step, next_x + step, step))

            for xh in xh_range:
                pos = (xh, y)
                if (pos not in obstacles or obstacles[pos] != "wall") and (pos not in body_positions):
                    path.append(pos)
    return path

def update_obstacles(sight, body):
    """
    Atualiza a estrutura de dados com obstáculos detectados (paredes e corpos de outras cobras).
    """
    global obstacles
    obstacles.clear()  # Limpa obstáculos anteriores
    for x_str, col in sight.items():
        for y_str, tile in col.items():
            x, y = int(x_str), int(y_str)
            pos = (x, y)

            if tile == 1:
                obstacles[pos] = "wall"  # Marca como parede

            elif tile == 4:
                obstacles[pos] = "body"  # Marca como corpo de outra cobra

    for b in body:
        obstacles[b] = "body"

def choose_new_direction(snake_body, body):
    global current_direction
    head = tuple(snake_body[0])  # Cabeça da cobra

    directions = {
        "w": move_in_direction(head, 0, can_traverse),  # Norte
        "a": move_in_direction(head, 3, can_traverse),  # Oeste
        "s": move_in_direction(head, 2, can_traverse),  # Sul
        "d": move_in_direction(head, 1, can_traverse),  # Leste
    }

    if current_direction:
        directions.pop(opposite_direction(current_direction), None)

    classified_directions = {dir: classify_position(pos, body) for dir, pos in directions.items()}
    #print(f"Classificação das direções: {classified_directions}")

    for classification in ["possible", "dangerous"]:
        for dir, cls in classified_directions.items():
            if cls == classification:
                current_direction = dir
                return dir
    return None

def classify_position(position, body):
    """
    Classifica a posição como "not possible", "very dangerous", "dangerous" ou "possible".
    """
    if position in body or obstacles.get(position) == "wall":
        return "not possible"
    
    if obstacles.get(position) == "body":
        return "very dangerous"
    
    accessible_area = get_accessible_area(position, body)
    snake_length = len(body)
    
    # Penalidade alta para áreas muito pequenas
    if accessible_area < snake_length * 1.5:
        return "very dangerous"
    
    # Verifica se há vizinhos seguros
    neighbors = get_neighbors(position, (MAP_WIDTH, MAP_HEIGHT), obstacles, can_traverse)
    safe_neighbors = [n for n in neighbors if n not in body and obstacles.get(n) != "wall"]
    if len(safe_neighbors) < 2:
        return "dangerous"
    
    return "possible"

def get_accessible_area(start_position, body):
    """
    Calcula a área acessível a partir da posição inicial, considerando os obstáculos e o corpo da cobra.
    """
    visited = set()
    queue = [start_position]
    accessible_area = 0

    while queue:
        current = queue.pop(0)
        if current in visited or current in body or obstacles.get(current) == "wall":
            continue
        visited.add(current)
        accessible_area += 1

        neighbors = get_neighbors(current, (MAP_WIDTH, MAP_HEIGHT), obstacles, can_traverse)
        for neighbor in neighbors:
            if neighbor not in visited and neighbor not in body:
                queue.append(neighbor)

    return accessible_area

def is_dead_end_with_penalty(position, body):
    """
    Verifica se a posição leva a um beco sem saída e retorna uma penalidade.
    """
    accessible_area = get_accessible_area(position, body)
    snake_length = len(body)
    
    # Penalidade alta para áreas muito pequenas
    if accessible_area < snake_length * 1.2:
        return True, accessible_area
    # Penalidade moderada para áreas perigosas
    elif accessible_area < snake_length * 1.5:
        return False, (snake_length - accessible_area) * 0.5
    return False, 0

def is_move_safe(position, body):
    """
    Verifica se o movimento para a posição especificada é seguro.
    """
    classification = classify_position(position, body)
    if classification == "não possível":
        #print(f"Movimento para {position} não é seguro: {classification}.")
        return False
    elif classification == "perigosa":
        #print(f"Movimento para {position} é perigoso: {classification}.")
        return True
    else:
        #print(f"Movimento para {position} é seguro: {classification}.")
        return True

def move_in_direction(position, direction, can_traverse=True):
    """
    Move a cobra na direção especificada.
    Quando can_traverse é False, não atravessa o mapa.
    """
    x, y = position
    
    if direction == 0:  # Norte
        y -= 1
        if can_traverse:
            y %= MAP_HEIGHT
    elif direction == 1:  # Leste
        x += 1
        if can_traverse:
            x %= MAP_WIDTH
    elif direction == 2:  # Sul
        y += 1
        if can_traverse:
            y %= MAP_HEIGHT
    elif direction == 3:  # Oeste
        x -= 1
        if can_traverse:
            x %= MAP_WIDTH

    return (x, y)

def direction_from_steps(head, next_step, can_traverse=True):
    """
    Retorna a direção do movimento a partir da posição atual até o próximo passo, considerando o wrap-around do mapa.
    """
    x1, y1 = head
    x2, y2 = next_step
    width, height = MAP_WIDTH, MAP_HEIGHT

    if can_traverse:
        dx = (x2 - x1) % width
        dy = (y2 - y1) % height

        if dy == 0:
            # Movimento horizontal
            if dx == 1:
                return "d"  # Leste
            elif dx == (width - 1):
                return "a"  # Oeste (wrap-around)
        elif dx == 0:
            # Movimento vertical
            if dy == 1:
                return "s"  # Sul
            elif dy == (height - 1):
                return "w"  # Norte (wrap-around)
    else:
        # Sem wrap-around; cálculos diretos
        if x2 > x1:
            return "d"  # Leste
        elif x2 < x1:
            return "a"  # Oeste
        elif y2 > y1:
            return "s"  # Sul
        elif y2 < y1:
            return "w"  # Norte
    return None  # Caso contrário

def is_opposite_direction(dir1, dir2):
    """
    Verifica se dir2 é a direção oposta de dir1.
    """
    return dir2 == opposite_direction(dir1)

def predict_other_snake_moves(sight):
    """
    Previsão simples dos movimentos das outras cobras com base na visão atual.
    """
    predicted_positions = set()
    for x_str, col in sight.items():
        for y_str, tile in col.items():
            if tile == 4:  # Parte do corpo de outra cobra
                x, y = int(x_str), int(y_str)
                neighbors = get_neighbors((x, y), (MAP_WIDTH, MAP_HEIGHT), obstacles, can_traverse)
                for neighbor in neighbors:
                    predicted_positions.add(neighbor)
    return predicted_positions

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    SERVER = os.environ.get("SERVER", "localhost")
    PORT = os.environ.get("PORT", "8001")
    NAME = os.environ.get("NAME", getpass.getuser())
    loop.run_until_complete(agent_loop(f"{SERVER}:{PORT}", NAME))

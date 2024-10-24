import signal
import pygame

from embodied.envs.minetest_gymnasium import BoadConfig, MinetestGymnasium

KEY_TO_ACTION_INDEX = dict([
    (pygame.K_w, 0),
    (pygame.K_a, 1),
    (pygame.K_s, 2),
    (pygame.K_d, 3),
    (pygame.K_j, 4),
    (pygame.K_UP, 8),
    (pygame.K_DOWN, 7),
    (pygame.K_LEFT, 6),
    (pygame.K_RIGHT, 5),
])

def game_loop():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.key in KEY_TO_ACTION_INDEX:
                    state, reward, terminated, truncated, info = env.step(KEY_TO_ACTION_INDEX[event.key])
                    health = state["health"].item()
                    food = state["food"].item()
                    water = state["water"].item()

                    print(f"reward: {reward:3.1f}, health: {health:2.0f}, food: {food:4.0f}, water: {water:4.0f}")
                    if terminated:
                        running = False
                        break

            env.render()

def sigint_handler(_sig, _frame):
    env.close()
    pygame.quit()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)

    pygame.init()

    config: BoadConfig = {
        'food_consumption_per_second': 100,
        'water_consumption_per_second': 100,
    }

    with MinetestGymnasium("boad", screen_size=1024, config=config, render_mode='human') as env:
        env.reset()
        game_loop()

    pygame.quit()

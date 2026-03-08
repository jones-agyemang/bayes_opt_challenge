PROPOSE_NEXT     = "propose_next"
PROPOSE_NEXT_RND = "propose_next_rnd_sampling"
BASE_CYCLE_PARAMETERS = {
    i: {
        f"function_{func}": {
            "acquisition": {
                "strategy": "ucb",
                "params": { "kappa": 2.0 }
            },
            "proposer": PROPOSE_NEXT
        }
        for func in range(1, 9)
    }
    for i in range(1, 5)
}

DYN_CYCLE_PARAMETERS = {
    5: {
        "function_1": {
            "acquisition": {
                "strategy": "ei",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT_RND
        },
        "function_2": {
            "acquisition": {
                "strategy": "ei",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT_RND
        },
        "function_3": {
            "acquisition": {
                "strategy": "ei",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT_RND
        },
        "function_4": {
            "acquisition": {
                "strategy": "ei",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT
        },
        "function_5": {
            "acquisition": {
                "strategy": "ei",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT
        },
        "function_6": {
            "acquisition": {
                "strategy": "ei",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT
        },
        "function_7": {
            "acquisition": {
                "strategy": "ei",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT
        },
        "function_8": {
            "acquisition": {
                "strategy": "ei",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT
        }
    },
    6: {
        "function_1": {
            "acquisition": {
                "strategy": "ucb",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT_RND
        },
        "function_2": {
            "acquisition": {
                "strategy": "ucb",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT_RND
        },
        "function_3": {
            "acquisition": {
                "strategy": "ucb",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT_RND
        },
        "function_4": {
            "acquisition": {
                "strategy": "ucb",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT
        },
        "function_5": {
            "acquisition": {
                "strategy": "ucb",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT
        },
        "function_6": {
            "acquisition": {
                "strategy": "ucb",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT
        },
        "function_7": {
            "acquisition": {
                "strategy": "ucb",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT
        },
        "function_8": {
            "acquisition": {
                "strategy": "ucb",
                "params": { "kappa": 25 }
            },
            "proposer": PROPOSE_NEXT
        }
    }
}

def get_cycle_parameters(): return {**BASE_CYCLE_PARAMETERS, **DYN_CYCLE_PARAMETERS}

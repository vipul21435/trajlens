"""An OpenHands action never borrows the observation of a later action."""

from trajlens.adapters.openhands import parse_openhands


def test_message_does_not_take_run_output():
    events = [
        {"id": 1, "source": "user", "action": "message", "args": {"content": "fix it"}},
        {"id": 2, "source": "agent", "action": "message", "args": {"content": "on it"}},
        {"id": 3, "source": "agent", "action": "run", "args": {"command": "pytest"}},
        {
            "id": 4,
            "source": "agent",
            "observation": "run",
            "cause": 3,
            "content": "1 failed",
            "extras": {"exit_code": 1},
        },
        {"id": 5, "source": "agent", "action": "finish", "args": {}},
    ]
    traj = parse_openhands(events, "oh")
    assert [s.raw_action for s in traj.steps] == ["message", "run", "finish"]
    assert traj.steps[0].exit_code is None and traj.steps[0].observation == ""
    assert traj.steps[1].exit_code == 1 and traj.steps[1].observation == "1 failed"


def test_uncaused_observation_pairs_only_with_the_action_before_it():
    events = [
        {"id": 1, "source": "agent", "action": "run", "args": {"command": "ls"}},
        {
            "id": 2,
            "source": "agent",
            "observation": "run",
            "content": "a.py",
            "extras": {"exit_code": 0},
        },
        {"id": 3, "source": "agent", "action": "run", "args": {"command": "ls b"}},
        {"id": 4, "source": "agent", "action": "run", "args": {"command": "ls c"}},
        {"id": 5, "source": "agent", "observation": "run", "cause": 4, "content": "c"},
    ]
    traj = parse_openhands(events, "oh")
    assert [s.observation for s in traj.steps] == ["a.py", "", "c"]

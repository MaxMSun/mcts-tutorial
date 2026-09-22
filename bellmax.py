from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class Tree(NamedTuple):
    state: Any
    children: jax.Array
    parent: jax.Array
    action: jax.Array
    reward: jax.Array
    depth: jax.Array
    visits: jax.Array
    value_sum: jax.Array
    size: jax.Array


class SearchResult(NamedTuple):
    actions: jax.Array
    action_visits: jax.Array
    action_values: jax.Array
    node_states: Any
    node_visits: jax.Array
    q_values: jax.Array
    tree: Tree


class _Selection(NamedTuple):
    node: jax.Array
    path: jax.Array
    length: jax.Array


class MCTS:
    def __init__(
        self,
        *,
        max_depth: int,
        num_sim: int,
        discount: float = 1.0,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least one")
        if num_sim < 1:
            raise ValueError("num_sim must be at least one")
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        self.max_depth = max_depth
        self.num_sim = num_sim
        self.discount = discount

    def actions(self, state):
        raise NotImplementedError

    def transition(self, state, action):
        raise NotImplementedError

    def reward(self, state, action, next_state, *args):
        raise NotImplementedError

    def terminal(self, state, *args):
        return jnp.asarray(False)

    def leaf_value(self, state, *args):
        return jnp.asarray(0.0, dtype=jnp.float32)

    def select(self, key, state, actions, visits, values):
        exploration = 1.4
        scores = values + exploration * jnp.sqrt(
            jnp.log(jnp.sum(visits) + 1.0) / visits
        )
        return jnp.argmax(scores)

    def expand(self, key, state, actions):
        return jax.random.randint(key, (), 0, len(actions))

    def rollout(self, key, state, actions):
        return jax.random.randint(key, (), 0, len(actions))

    def _state(self, tree, node):
        return jax.tree.map(lambda values: values[node], tree.state)

    def _set_state(self, storage, node, state):
        return jax.tree.map(lambda values, value: values.at[node].set(value), storage, state)

    def _init_tree(self, state, actions):
        capacity = self.num_sim + 1
        num_actions = actions.shape[0]
        state_storage = jax.tree.map(
            lambda value: jnp.zeros(
                (capacity,) + jnp.asarray(value).shape,
                dtype=jnp.asarray(value).dtype,
            ).at[0].set(value),
            state,
        )
        return Tree(
            state=state_storage,
            children=jnp.full((capacity, num_actions), -1, dtype=jnp.int32),
            parent=jnp.full(capacity, -1, dtype=jnp.int32),
            action=jnp.zeros((capacity,) + actions.shape[1:], dtype=actions.dtype),
            reward=jnp.zeros(capacity, dtype=jnp.float32),
            depth=jnp.zeros(capacity, dtype=jnp.int32),
            visits=jnp.zeros(capacity, dtype=jnp.int32),
            value_sum=jnp.zeros(capacity, dtype=jnp.float32),
            size=jnp.asarray(1, dtype=jnp.int32),
        )

    def _select(self, key, tree, args):
        path = jnp.full(self.max_depth + 1, -1, dtype=jnp.int32).at[0].set(0)
        initial = _Selection(
            node=jnp.asarray(0, dtype=jnp.int32),
            path=path,
            length=jnp.asarray(1, dtype=jnp.int32),
        )

        def condition(selection):
            node = selection.node
            state = self._state(tree, node)
            fully_expanded = jnp.all(tree.children[node] >= 0)
            return (
                fully_expanded
                & ~self.terminal(state, *args)
                & (tree.depth[node] < self.max_depth)
            )

        def step(selection):
            node = selection.node
            state = self._state(tree, node)
            actions = jnp.asarray(self.actions(state))
            children = tree.children[node]
            visits = tree.visits[children]
            values = tree.value_sum[children] / visits
            choice = self.select(
                jax.random.fold_in(key, selection.length),
                state,
                actions,
                visits,
                values,
            )
            child = children[choice]
            return _Selection(
                node=child,
                path=selection.path.at[selection.length].set(child),
                length=selection.length + 1,
            )

        return jax.lax.while_loop(condition, step, initial)

    def _expand(self, key, tree, selection, args):
        parent = selection.node
        state = self._state(tree, parent)
        actions = jnp.asarray(self.actions(state))
        available = tree.children[parent] < 0
        proposed = self.expand(key, state, actions)
        fallback_scores = jax.random.uniform(
            jax.random.fold_in(key, 1),
            (actions.shape[0],),
        )
        fallback = jnp.argmax(jnp.where(available, fallback_scores, -jnp.inf))
        action_index = jnp.where(available[proposed], proposed, fallback)
        can_expand = (
            jnp.any(available)
            & ~self.terminal(state, *args)
            & (tree.depth[parent] < self.max_depth)
            & (tree.size < self.num_sim + 1)
        )

        def add(current):
            node = current.size
            action = actions[action_index]
            next_state = self.transition(state, action)
            reward = self.reward(state, action, next_state, *args)
            return current._replace(
                state=self._set_state(current.state, node, next_state),
                children=current.children.at[parent, action_index].set(node),
                parent=current.parent.at[node].set(parent),
                action=current.action.at[node].set(action),
                reward=current.reward.at[node].set(reward),
                depth=current.depth.at[node].set(current.depth[parent] + 1),
                size=current.size + 1,
            )

        tree = jax.lax.cond(can_expand, add, lambda current: current, tree)
        node = jnp.where(can_expand, tree.size - 1, parent)
        path = jax.lax.cond(
            can_expand,
            lambda value: value.at[selection.length].set(node),
            lambda value: value,
            selection.path,
        )
        length = selection.length + can_expand.astype(jnp.int32)
        return tree, _Selection(node=node, path=path, length=length)

    def _rollout(self, key, state, args, remaining_depth):
        class Carry(NamedTuple):
            state: Any
            value: jax.Array
            weight: jax.Array
            active: jax.Array

        initial = Carry(
            state=state,
            value=jnp.asarray(0.0, dtype=jnp.float32),
            weight=jnp.asarray(1.0, dtype=jnp.float32),
            active=~self.terminal(state, *args),
        )

        def step(carry, index):
            active = carry.active & (index < remaining_depth)
            actions = jnp.asarray(self.actions(carry.state))
            action_index = self.rollout(
                jax.random.fold_in(key, index),
                carry.state,
                actions,
            )
            action = actions[action_index]
            next_state = jax.lax.cond(
                active,
                lambda _: self.transition(carry.state, action),
                lambda _: carry.state,
                operand=None,
            )
            reward = jax.lax.cond(
                active,
                lambda _: self.reward(carry.state, action, next_state, *args),
                lambda _: jnp.asarray(0.0, dtype=jnp.float32),
                operand=None,
            )
            return Carry(
                state=next_state,
                value=carry.value + carry.weight * reward,
                weight=carry.weight * jnp.where(active, self.discount, 1.0),
                active=active & ~self.terminal(next_state, *args),
            ), None

        final, _ = jax.lax.scan(step, initial, jnp.arange(self.max_depth))
        leaf = jnp.where(
            ~self.terminal(final.state, *args),
            self.leaf_value(final.state, *args),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return final.value + final.weight * leaf

    def _backup(self, tree, selection, leaf_value):
        positions = jnp.arange(self.max_depth, 0, -1)

        def step(carry, position):
            current, value = carry
            active = position < selection.length
            node = jnp.where(active, selection.path[position], 0)
            updated_value = current.reward[node] + self.discount * value
            value = jnp.where(active, updated_value, value)
            visits = current.visits.at[node].add(active.astype(jnp.int32))
            value_sum = current.value_sum.at[node].add(
                jnp.where(active, value, 0.0)
            )
            return (current._replace(visits=visits, value_sum=value_sum), value), None

        (tree, value), _ = jax.lax.scan(
            step,
            (tree, leaf_value),
            positions,
        )
        tree = tree._replace(
            visits=tree.visits.at[0].add(1),
            value_sum=tree.value_sum.at[0].add(value),
        )
        return tree

    def _simulate(self, key, tree, args):
        selection = self._select(jax.random.fold_in(key, 0), tree, args)
        tree, selection = self._expand(
            jax.random.fold_in(key, 1),
            tree,
            selection,
            args,
        )
        state = self._state(tree, selection.node)
        remaining_depth = self.max_depth - tree.depth[selection.node]
        value = self._rollout(
            jax.random.fold_in(key, 2),
            state,
            args,
            remaining_depth,
        )
        return self._backup(tree, selection, value)

    @partial(jax.jit, static_argnums=0)
    def search(self, key, state, *args):
        actions = jnp.asarray(self.actions(state))
        tree = self._init_tree(state, actions)

        def simulate(index, current):
            return self._simulate(jax.random.fold_in(key, index), current, args)

        tree = jax.lax.fori_loop(0, self.num_sim, simulate, tree)
        children = tree.children[0]
        valid = children >= 0
        safe_children = jnp.maximum(children, 0)
        visits = jnp.where(valid, tree.visits[safe_children], 0)
        values = jnp.where(
            valid,
            tree.value_sum[safe_children] / jnp.maximum(tree.visits[safe_children], 1),
            jnp.nan,
        )
        return SearchResult(
            actions=actions,
            action_visits=visits,
            action_values=values,
            node_states=tree.state,
            node_visits=tree.visits,
            q_values=tree.value_sum / jnp.maximum(tree.visits, 1),
            tree=tree,
        )

import numpy as np
import matplotlib.pyplot as plt

class DecisionTreeVisualizer:
    """
    Standalone visualization class for your existing decision tree structure.
    
    Attributes:
        tree (dict): The root node of the decision tree, represented as a nested dictionary.
        id2node (dict): Mapping of Python object IDs to corresponding node dictionaries.
        depth (int): Depth (height) of the tree, typically computed during training or pruning.
    """

    def __init__(self, tree:dict=None, id2node:dict=None, depth:int=None):
        """Initializes the DecisionTreeVisualizer.

        Args:
            tree (dict, optional): Root dictionary representing the decision tree.
                Must contain 'leaf', 'label', 'left', 'right', etc.
            id2node (dict, optional): Mapping of id(node) to node dict for fast access.
            depth (int, optional): Precomputed depth of the tree (faster than recomputing).
        """
        self.tree = tree
        self.id2node = id2node
        self.depth = depth
        
    def update(self, tree:dict=None, id2node:dict=None, depth:int=None)->None:
        """ 
        Update class attributes according to any values given.

        Args:
            tree (dict): a nested dictionary that represents a decision tree 
            id2node (dict)
            depth (int)
        """
        if tree:
            self.tree = tree
            if id2node==None:
                self.update_nodes()
            if depth==None:
                self._depth(self.tree)
        if id2node:
            self.id2node = id2node
        if depth:
            self.depth = depth
        
        if tree==None and id2node==None and depth==None:
            self.update_nodes()
            self._depth(self.tree)


    def update_nodes(self):
        """
        Update the collection of nodes manually
        """
        self.id2node = {}
        self._collect_nodes(self.tree, self.id2node)


    def plot(self, save_path:str=None, num_lvl:int=8, ax=None, fontsize:int=8, node_pad:float=0.3):
        """Plot the decision tree with non-overlapping layout.

        Args:
            save_path (str | None): If set, save plot to this path.
            ax : Optional existing Axes to draw on.
            fontsize (int): Text font size for nodes.
            node_pad (float): Padding inside node boxes.

        Returns:
            matplotlib.axes.Axes: Axes with the plotted tree.
        """
        num_lvl = min(num_lvl, self.depth)
        pos, edges, truncated = self._assign_positions(self.tree, max_depth=num_lvl)
        xs, ys = self._normalize_positions(pos)

        root_ax = ax is None
        if root_ax:
            fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(pos)), max(6, 0.8 * self.depth)))
            ax.axis("off")
        else:
            fig = ax.figure

        # draw edges
        for u, v in edges:
            ax.plot([xs[u], xs[v]], [ys[u], ys[v]], lw=0.8)

        # draw nodes
        for nid in pos:
            node = self.id2node[nid]
            x, y = xs[nid], ys[nid]
            if node["leaf"]:
                txt = f"Leaf\nLabel:{node['label']}"
                fc = 'lightgreen'
            else:
                txt = f"[X{node['attr']} < {node['value']:.3f}]"
                fc = 'lightblue'
            ax.text(x, y, txt, ha='center', va='center', fontsize=fontsize,
                    bbox=dict(boxstyle='round,pad='+str(node_pad), facecolor=fc, alpha=0.7))
        
        # draw continuation *two* edges (left & right)
        branch_dx = 0.04   # horizontal offset
        branch_dy = 0.06   # vertical downward drop

        for nid in truncated:
            x, y = xs[nid], ys[nid]

            # left child continuation
            ax.plot([x, x - branch_dx], [y, y - branch_dy],
                    linestyle='--', linewidth=1.2, color='0.3', alpha=0.8)

            # right child continuation
            ax.plot([x, x + branch_dx], [y, y - branch_dy],
                    linestyle='--', linewidth=1.2, color='0.3', alpha=0.8)

            # small text hint (optional)
            ax.text(x, y - branch_dy - 0.015, "…", ha='center', va='top', fontsize=fontsize)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            if root_ax: plt.close(fig)

        return ax

    # ---------- Helper functions ----------
    @staticmethod
    def _collect_nodes(node:dict, id2node:dict):
        """Recursively create id2node dictioanry from a node 

        Args:
            node (dict): A nested dictionary of decision tree 
            id2node (dict): A dictionary of decision tree node index-value pair
        """
        id2node[id(node)] = node
        if not node["leaf"]:
            DecisionTreeVisualizer._collect_nodes(node["left"], id2node)
            DecisionTreeVisualizer._collect_nodes(node["right"], id2node)

    @staticmethod
    def _depth(node:dict)->int:
        """Recursively calculate the depth of the tree

        Args:
            node (dict): A nested dictionary of decision tree 

        Returns:
            int: depth
        """
        if node["leaf"]: return 1
        return 1 + max(DecisionTreeVisualizer._depth(node["left"]),
                       DecisionTreeVisualizer._depth(node["right"]))

    @staticmethod
    def _assign_positions(node, max_depth:int=None, depth:int=0,
                        x_cursor:list[int]=None, pos:dict=None,
                        edges:list=None, truncated:set=None)->tuple[dict, list, set]:
        """Assign raw (x, y) positions for each tree node using in-order layout.
        Returns (pos, edges, truncated) where truncated contains ids of nodes
        that are internal but truncated at max_depth.
        """
        if x_cursor is None: x_cursor = [0]
        if pos is None: pos = {}
        if edges is None: edges = []
        if truncated is None: truncated = set()

        nid = id(node)

        if node["leaf"]:
            pos[nid] = (x_cursor[0], depth)
            x_cursor[0] += 1
            return pos, edges, truncated

        # If we've hit the plotting depth, place node but don't expand children.
        if max_depth and depth == max_depth:
            pos[nid] = (x_cursor[0], depth)
            x_cursor[0] += 1
            # Mark as truncated if it's not a leaf in the real tree
            truncated.add(nid)
            return pos, edges, truncated

        # Recurse children
        pos, edges, truncated = DecisionTreeVisualizer._assign_positions(
            node["left"], max_depth, depth+1, x_cursor, pos, edges, truncated)
        pos, edges, truncated = DecisionTreeVisualizer._assign_positions(
            node["right"], max_depth, depth+1, x_cursor, pos, edges, truncated)

        xl, _ = pos[id(node["left"])]
        xr, _ = pos[id(node["right"])]
        pos[nid] = ((xl + xr) / 2.0, depth)
        edges.append((nid, id(node["left"])))
        edges.append((nid, id(node["right"])))
        return pos, edges, truncated


    @staticmethod
    def _normalize_positions(pos:dict)->dict:
        """Normalizes raw (x, y) layout positions to a 0–1 range for plotting.

        This rescales:
        - x-coordinates to [0, 1] where 0 = leftmost leaf, 1 = rightmost leaf
        - y-coordinates to [0, 1] where 1 = root (top) and 0 = lowest leaf
        
        This ensures proper scaling for clean, non-overlapping visualization
        regardless of absolute tree size.

        Args:
            pos (dict[int, tuple[float, float]]):
                A dictionary mapping `id(node)` to its raw `(x, y)` layout coordinates
                as computed during `_assign_positions()`.

        Returns:
            tuple[dict[int, float], dict[int, float]]:
                Two dictionaries:
                - normalized_x: mapping id(node) → normalized x ∈ [0, 1]
                - normalized_y: mapping id(node) → normalized y ∈ [0, 1]
        """
        node_ids = list(pos.keys())
        xs = np.array([pos[nid][0] for nid in node_ids], float)
        ys = np.array([pos[nid][1] for nid in node_ids], float)

        if np.ptp(xs) > 0: xs = (xs - xs.min()) / np.ptp(xs)
        else: xs[:] = 0.5

        if np.ptp(ys) > 0: ys = 1.0 - (ys - ys.min()) / np.ptp(ys)
        else: ys[:] = 0.5

        return {nid: xs[i] for i, nid in enumerate(node_ids)}, \
               {nid: ys[i] for i, nid in enumerate(node_ids)}

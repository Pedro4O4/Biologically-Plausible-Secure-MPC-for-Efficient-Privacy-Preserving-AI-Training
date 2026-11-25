"""
Phase 5: three-party replicated secret sharing, simulated step by step in plain Python.

This is a teaching model of the protocol family we will use in MP-SPDZ
(replicated-ring-party.x: 3 parties, honest majority, semi-honest). It is NOT a secure
implementation (one process plays all three parties); its job is to show, with real
numbers, what every party holds and sends in every step, and to count the cost.

Sharing. A secret x in the ring Z_(2^k) is split into three random shares with
x0 + x1 + x2 = x (mod 2^k). Party i holds TWO of them: (x_i, x_(i+1)).
    * addition                : every party adds its own shares. No communication.
    * multiplication          : local products + a random mask + ONE message per party.
    * comparison (a < b)      : sign of (a - b) -> bit operations (AND = one round each).

Boolean sharing (used by the comparison) is the same idea with XOR instead of +,
and AND instead of x.
"""

import random

_SUB = "₀₁₂"          # subscript 0 1 2


def sub(i):
    return _SUB[i % 3]


class Shared:
    """A secret split into three shares. kind: 'A' (arithmetic, mod 2^k) or 'B' (boolean, XOR)."""

    __slots__ = ("sh", "kind", "name", "w")

    def __init__(self, sh, kind, name, w):
        self.sh, self.kind, self.name, self.w = list(sh), kind, name, w

    def view(self, i):
        """What party i holds: its own share and the next one."""
        return self.sh[i % 3], self.sh[(i + 1) % 3]


class MPC:
    SIZES = (8, 16, 32, 64)

    def __init__(self, bits=8, seed=0):
        if bits not in self.SIZES:
            raise ValueError(f"ring size must be one of {self.SIZES}")
        self.k, self.mod = bits, 1 << bits
        self.rng = random.Random(seed)
        self.steps = []
        self.cost = {p: {"rounds": 0, "messages": 0, "bits": 0} for p in ("input", "op", "reveal")}
        self.and_bits = 0
        self.round_no = 0

    # ---------- number formatting ----------
    def u(self, v):
        return str(v % self.mod)

    def si(self, v):
        v %= self.mod
        return v - self.mod if v >= self.mod // 2 else v

    def sg(self, v):
        return str(self.si(v))

    def bn(self, v, w=None):
        w = w or self.k
        t = format(v & ((1 << w) - 1), f"0{w}b")
        return " ".join(t[i:i + 4] for i in range(0, len(t), 4))

    def fmt(self, X, v):
        return self.u(v) if X.kind == "A" else self.bn(v, X.w)

    # ---------- bookkeeping ----------
    def _round(self, phase, new=True):
        if new:
            self.cost[phase]["rounds"] += 1
            self.round_no += 1
        return self.round_no

    def _sent(self, phase, bits):
        self.cost[phase]["messages"] += 1
        self.cost[phase]["bits"] += bits

    def _step(self, section, title, why, how=(), parties=None, msgs=(), god=(), rnd=None):
        self.steps.append({
            "n": len(self.steps), "section": section, "title": title, "why": why,
            "how": list(how), "round": rnd, "local": rnd is None,
            "parties": parties or [{"holds": [], "does": []} for _ in range(3)],
            "msgs": list(msgs), "god": list(god),
        })

    def _holds(self, i, *items):
        """items: (Shared, hot). Rows for the two shares party i holds of each variable."""
        rows = []
        for X, hot in items:
            for j in (i, (i + 1) % 3):
                rows.append({"k": f"{X.name}{sub(j)}", "v": self.fmt(X, X.sh[j % 3]), "hot": hot})
        return rows

    def _secret(self, X):
        if X.kind == "A":
            return sum(X.sh) % self.mod
        r = 0
        for s in X.sh:
            r ^= s
        return r

    def _ok(self, cond, what):
        if not cond:
            raise AssertionError("simulation invariant broken: " + what)
        return "✓"

    # ---------- steps of the protocol ----------
    def setup(self, a, b, op_text):
        k = self.k
        sg_lo, sg_hi = -(self.mod // 2), self.mod // 2 - 1
        parties = []
        for i in range(3):
            holds = [{"k": "keys", "v": f"K{sub(i)}, K{sub(i + 1)}", "hot": False}]
            if i == 0:
                holds.append({"k": "private a", "v": str(a), "hot": True})
            if i == 1:
                holds.append({"k": "private b", "v": str(b), "hot": True})
            parties.append({"holds": holds, "does": []})
        self._step(
            "setup", "Who knows what at the start",
            f"Three computers (parties) P₀, P₁, P₂ want to compute {op_text} together. P₀ owns the number a and P₁ owns b. "
            "Nobody may learn the other party's number, and no single party may learn the result's inputs even while computing.",
            [f"All numbers live in a ring: Z_{self.mod} (k = {k} bits). Going past {self.mod - 1} wraps around to 0, like an {k}-bit counter.",
             f"Signed reading: 0..{sg_hi} mean themselves, {sg_hi + 1}..{self.mod - 1} mean {sg_lo}..-1.",
             f"MP-SPDZ uses k = 64. We use k = {k} here only so the numbers stay readable.",
             "Setup, done once before any data exists: neighbours agree on secret keys. K₀ is known to P₀ and P₂, "
             "K₁ to P₁ and P₀, K₂ to P₂ and P₁. So P_i knows (K_i, K_i+1). "
             "A shared key lets two parties produce the same random number without sending anything."],
            parties, god=[f"True inputs: a = {a}, b = {b}. Only this simulation can see both."])

    def share_input(self, owner, value, name, new_round=True):
        k, mod, o = self.k, self.mod, owner
        x = [0, 0, 0]
        x[o] = self.rng.randrange(mod)                       # F(K_o): known to owner and party o-1
        x[(o + 1) % 3] = self.rng.randrange(mod)             # F(K_o+1): known to owner and party o+1
        x[(o + 2) % 3] = (value - x[o] - x[(o + 1) % 3]) % mod
        X = Shared(x, "A", name, k)
        third = (o + 2) % 3
        rnd = self._round("input", new_round)
        parties = []
        for i in range(3):
            does = []
            if i == o:
                does = [f"{name}{sub(o)} = F(K{sub(o)}) = {x[o]}   (random, from the key shared with P{sub(o - 1)})",
                        f"{name}{sub(o + 1)} = F(K{sub(o + 1)}) = {x[(o + 1) % 3]}   (random, from the key shared with P{sub(o + 1)})",
                        f"{name}{sub(third)} = {name} - {name}{sub(o)} - {name}{sub(o + 1)} = {value} - {x[o]} - {x[(o + 1) % 3]} (mod {mod}) = {x[third]}",
                        f"send {name}{sub(third)} to P{sub(o + 1)} and P{sub(o + 2)}"]
            elif i == (o + 1) % 3:
                does = [f"{name}{sub(i)} = F(K{sub(i)}) = {x[i]}   (same random number as the owner, from the shared key)",
                        f"receive {name}{sub(third)} = {x[third]} from P{sub(o)}"]
            else:
                does = [f"{name}{sub(o)} = F(K{sub(o)}) = {x[o]}   (from the key shared with the owner)",
                        f"receive {name}{sub(third)} = {x[third]} from P{sub(o)}"]
            parties.append({"holds": self._holds(i, (X, True)), "does": does})
        msgs = []
        for d in ((o + 1) % 3, (o + 2) % 3):
            msgs.append({"src": o, "dst": d, "label": f"{name}{sub(third)}", "value": str(x[third]), "bits": k})
            self._sent("input", k)
        c = self._ok(sum(x) % mod == value % mod, "shares must add up to the secret")
        self._step(
            "input", f"P{sub(o)} secret-shares {name} = {value}",
            f"To hide {name}, P{sub(o)} splits it into three random-looking pieces {name}{sub(0)} + {name}{sub(1)} + {name}{sub(2)} = {name} (mod {mod}). "
            f"Each party keeps two of the three pieces: P_i holds ({name}_i, {name}_i+1). A single party is always missing one piece, "
            f"so what it sees is random noise and tells it nothing about {name}.",
            [f"Two pieces are generated from keys, so they cost no communication.",
             f"The third piece is whatever makes the sum right. P{sub(o)} sends it to the two parties that need it (1 round).",
             f"Cost: 2 messages of {k} bits."],
            parties, msgs, [f"Check: {name}₀ + {name}₁ + {name}₂ = {x[0]} + {x[1]} + {x[2]} = {sum(x) % mod} (mod {mod}) = {name} {c}",
                            f"Signed reading of the secret: {self.sg(value)}"], rnd)
        return X

    def add(self, X, Y, name):
        mod = self.mod
        Z = Shared([(X.sh[j] + Y.sh[j]) % mod for j in range(3)], "A", name, self.k)
        parties = []
        for i in range(3):
            j = (i + 1) % 3
            parties.append({
                "holds": self._holds(i, (X, False), (Y, False), (Z, True)),
                "does": [f"{name}{sub(i)} = {X.name}{sub(i)} + {Y.name}{sub(i)} = {X.sh[i]} + {Y.sh[i]} = {Z.sh[i]}",
                         f"{name}{sub(j)} = {X.name}{sub(j)} + {Y.name}{sub(j)} = {X.sh[j]} + {Y.sh[j]} = {Z.sh[j]}"]})
        c = self._ok(sum(Z.sh) % mod == (self._secret(X) + self._secret(Y)) % mod, "sum shares")
        self._step(
            "add", f"Add: {name} = {X.name} + {Y.name}",
            "Addition is free. Every party adds, piece by piece, the two pieces it holds. No message is sent. "
            "Because (a₀+b₀) + (a₁+b₁) + (a₂+b₂) = (a₀+a₁+a₂) + (b₀+b₁+b₂) = a + b, the new pieces are a valid sharing of the sum.",
            ["Cost: 0 rounds, 0 messages, 0 bits.",
             "Same for: subtraction, adding a public constant, multiplying by a public constant."],
            parties, god=[f"Check: {name}₀+{name}₁+{name}₂ = {sum(Z.sh) % mod} = {X.name} + {Y.name} (mod {mod}) {c}",
                          f"Signed reading: {self.sg(self._secret(X))} + {self.sg(self._secret(Y))} = {self.sg(self._secret(Z))}"])
        return Z

    def _prod(self, X, Y, kind):
        """Local part of a product: raw z_i, masks, and masked output shares. Used by x (A) and AND (B)."""
        mod = self.mod
        z, r = [], [self.rng.randrange(mod) for _ in range(3)]
        for i in range(3):
            xi, xj, yi, yj = X.sh[i], X.sh[(i + 1) % 3], Y.sh[i], Y.sh[(i + 1) % 3]
            if kind == "A":
                z.append((xi * yi + xi * yj + xj * yi) % mod)
            else:
                z.append((xi & yi) ^ (xi & yj) ^ (xj & yi))
        if kind == "A":
            alpha = [(r[i] - r[(i + 1) % 3]) % mod for i in range(3)]
            out = [(z[i] + alpha[i]) % mod for i in range(3)]
        else:
            alpha = [r[i] ^ r[(i + 1) % 3] for i in range(3)]
            out = [z[i] ^ alpha[i] for i in range(3)]
        return z, r, alpha, out

    def mul(self, X, Y, name):
        mod, k = self.mod, self.k
        z, r, alpha, out = self._prod(X, Y, "A")
        Z = Shared(out, "A", name, k)
        xn, yn = X.name, Y.name
        secret = (self._secret(X) * self._secret(Y)) % mod

        # step 1: local products
        parties = []
        for i in range(3):
            j = (i + 1) % 3
            parties.append({
                "holds": self._holds(i, (X, False), (Y, False)),
                "does": [f"z{sub(i)} = {xn}{sub(i)}·{yn}{sub(i)} + {xn}{sub(i)}·{yn}{sub(j)} + {xn}{sub(j)}·{yn}{sub(i)}",
                         f"     = {X.sh[i]}·{Y.sh[i]} + {X.sh[i]}·{Y.sh[j]} + {X.sh[j]}·{Y.sh[i]}  (mod {mod})",
                         f"     = {z[i]}"]})
        c1 = self._ok(sum(z) % mod == secret, "z must add up to the product")
        self._step(
            "mul", f"Multiply (1/3): local products z_i for {name} = {xn} · {yn}",
            "Multiplying two secrets is harder than adding them: (a₀+a₁+a₂)(b₀+b₁+b₂) has nine cross terms, and each party only holds some of the pieces. "
            "The trick: party i multiplies the pieces it holds in a way that every cross term is covered by exactly one party. Together the three z_i add up to a·b.",
            [f"z_i = a_i·b_i + a_i·b_i+1 + a_i+1·b_i   (uses only what P_i already holds)",
             "Still no communication. But z_i is NOT safe to send yet: it is not uniformly random, it is a partial sum of the real product."],
            parties, god=[f"Check: z₀+z₁+z₂ = {z[0]} + {z[1]} + {z[2]} = {sum(z) % mod} = {xn}·{yn} = {secret} (mod {mod}) {c1}"])

        # step 2: masks
        parties = []
        for i in range(3):
            j = (i + 1) % 3
            parties.append({
                "holds": [{"k": f"z{sub(i)}", "v": str(z[i]), "hot": False}],
                "does": [f"r{sub(i)} = F(K{sub(i)}) = {r[i]},  r{sub(j)} = F(K{sub(j)}) = {r[j]}   (P{sub(i)} knows both keys)",
                         f"α{sub(i)} = r{sub(i)} - r{sub(j)} = {alpha[i]}",
                         f"{name}{sub(i)} = z{sub(i)} + α{sub(i)} = {z[i]} + {alpha[i]} = {out[i]}   (mod {mod})"]})
        c2 = self._ok(sum(alpha) % mod == 0, "masks must add to zero")
        c3 = self._ok(sum(out) % mod == secret, "masked shares must still add to the product")
        self._step(
            "mul", f"Multiply (2/3): hide the z_i with a zero-sum mask",
            "Each party adds a random-looking mask α_i to its z_i. The three masks are built from the shared keys so that α₀ + α₁ + α₂ = 0 "
            "(r₀-r₁ + r₁-r₂ + r₂-r₀ = 0). So the sum is unchanged, but every single value now looks uniformly random.",
            ["No communication: the masks come from keys the parties already share.",
             f"These masked values are the new pieces {name}₀, {name}₁, {name}₂ of the product."],
            parties, god=[f"Check: α₀+α₁+α₂ = {sum(alpha) % mod} (mod {mod}) {c2}",
                          f"Check: {name}₀+{name}₁+{name}₂ = {sum(out) % mod} = {xn}·{yn} {c3}"])

        # step 3: one message per party (reshare)
        rnd = self._round("op")
        parties, msgs = [], []
        for i in range(3):
            j = (i + 1) % 3
            prev = (i - 1) % 3
            parties.append({
                "holds": self._holds(i, (Z, True)),
                "does": [f"send {name}{sub(i)} = {out[i]} to P{sub(prev)}",
                         f"receive {name}{sub(j)} = {out[j]} from P{sub(j)}",
                         f"now P{sub(i)} holds ({name}{sub(i)}, {name}{sub(j)}) as the sharing format requires"]})
            msgs.append({"src": i, "dst": prev, "label": f"{name}{sub(i)}", "value": str(out[i]), "bits": k})
            self._sent("op", k)
        self._step(
            "mul", f"Multiply (3/3): each party sends one piece to its neighbour",
            "A sharing must have party i holding the pieces (c_i, c_i+1). Party i has c_i but not c_i+1, so each party sends its c_i to the previous party. "
            "That one message is the whole communication cost of a multiplication.",
            [f"Cost: 1 round, 3 messages of {k} bits = {3 * k} bits in total.",
             "Compare with addition: free. This is why the secret multiplications are what we count in MPC."],
            parties, msgs, [f"Result is a valid sharing of {xn}·{yn} = {secret} (signed: {self.sg(secret)})."], rnd)
        return Z

    def reveal(self, X, what):
        k = self.k
        rnd = self._round("reveal")
        w = X.w
        parties, msgs = [], []
        total = self._secret(X)
        for i in range(3):
            nxt, prv = (i + 1) % 3, (i - 1) % 3
            miss = (i + 2) % 3
            parties.append({
                "holds": self._holds(i, (X, False)),
                "does": [f"send {X.name}{sub(i)} to P{sub(nxt)}",
                         f"receive {X.name}{sub(miss)} = {self.fmt(X, X.sh[miss])} from P{sub(prv)}   (the piece P{sub(i)} was missing)",
                         (f"{what} = {X.name}₀ + {X.name}₁ + {X.name}₂ = {self.u(total)}" if X.kind == "A"
                          else f"{what} = {X.name}₀ ⊕ {X.name}₁ ⊕ {X.name}₂ = {self.bn(total, w)}")]})
            msgs.append({"src": i, "dst": nxt, "label": f"{X.name}{sub(i)}", "value": self.fmt(X, X.sh[i]), "bits": w})
            self._sent("reveal", w)
        self._step(
            "reveal", f"Open (reveal) {what}",
            "To output a value, every party needs the one piece it is missing. Party i receives it from its neighbour, then adds up all three pieces. "
            "This is the only moment a secret becomes public, and the parties choose it on purpose (here: the result only).",
            [f"Cost: 1 round, 3 messages of {w} bits.",
             "Revealing a value is a decision about what is allowed to leak. Intermediate values (like membrane potentials) must NOT be revealed."],
            parties, msgs, [f"Result: {self.u(total) if X.kind == 'A' else self.bn(total, w)}" + (f"  (signed: {self.sg(total)})" if X.kind == "A" else "")], rnd)
        return total

    # ---------- comparison ----------
    def _xor(self, X, Y, name):
        return Shared([x ^ y for x, y in zip(X.sh, Y.sh)], "B", name, X.w)

    def _shl(self, X, n, name):
        m = (1 << X.w) - 1
        return Shared([(s << n) & m for s in X.sh], "B", name, X.w)

    def _and_round(self, items, section, title, why, how, show=()):
        """items: (X, Y, out_name, meaningful_bits). One round: all ANDs in the list run in parallel."""
        rnd = self._round("op")
        outs, plines, msgs, god = [], [[], [], []], [], []
        bits_this_round = 0
        for X, Y, oname, nb in items:
            z, r, alpha, out = self._prod(X, Y, "B")
            Z = Shared(out, "B", oname, X.w)
            outs.append(Z)
            self.and_bits += nb
            bits_this_round += nb
            xn, yn = X.name, Y.name
            for i in range(3):
                j = (i + 1) % 3
                plines[i] += [f"{oname}: z{sub(i)} = ({xn}{sub(i)}∧{yn}{sub(i)}) ⊕ ({xn}{sub(i)}∧{yn}{sub(j)}) ⊕ ({xn}{sub(j)}∧{yn}{sub(i)}) = {self.bn(z[i])}",
                              f"{oname}: {oname}{sub(i)} = z{sub(i)} ⊕ α{sub(i)} = {self.bn(out[i])}   (mask α{sub(i)} = r{sub(i)} ⊕ r{sub(j)})"]
            c = self._ok(self._secret(Z) == (self._secret(X) & self._secret(Y)), "AND result")
            god.append(f"Check {oname}: {xn} ∧ {yn} = {self.bn(self._secret(X) & self._secret(Y))} = {oname}₀⊕{oname}₁⊕{oname}₂ {c}")
        parties = []
        for i in range(3):
            holds, seen = [], set()
            for row in ([r for X, Y, _, _ in items for r in self._holds(i, (X, False), (Y, False))]
                        + [r for V in show for r in self._holds(i, (V, False))]
                        + [r for Z in outs for r in self._holds(i, (Z, True))]):
                if row["k"] not in seen:
                    seen.add(row["k"])
                    holds.append(row)
            plines[i] += [f"send my {'/'.join(Z.name + sub(i) for Z in outs)} to P{sub(i - 1)}"]
            parties.append({"holds": holds, "does": plines[i]})
        for i in range(3):
            msgs.append({"src": i, "dst": (i - 1) % 3, "label": " + ".join(Z.name + sub(i) for Z in outs),
                         "value": " | ".join(self.bn(Z.sh[i]) for Z in outs), "bits": bits_this_round})
            self._sent("op", bits_this_round)
        self._step(section, title, why, how + [f"Cost of this round: 1 round, 3 messages, {3 * bits_this_round} bits ({bits_this_round} AND gates × 3)."],
                   parties, msgs, god, rnd)
        return outs

    def compare_lt(self, X, Y):
        """Secret bit [X < Y] for signed values, as a boolean sharing of one bit."""
        k, mod = self.k, self.mod
        m = k - 1
        sa, sb = self._secret(X), self._secret(Y)

        # C1: difference, local
        d = Shared([(X.sh[j] - Y.sh[j]) % mod for j in range(3)], "A", "d", k)
        parties = []
        for i in range(3):
            j = (i + 1) % 3
            parties.append({"holds": self._holds(i, (X, False), (Y, False), (d, True)),
                            "does": [f"d{sub(i)} = {X.name}{sub(i)} - {Y.name}{sub(i)} = {X.sh[i]} - {Y.sh[i]} = {d.sh[i]}",
                                     f"d{sub(j)} = {X.name}{sub(j)} - {Y.name}{sub(j)} = {X.sh[j]} - {Y.sh[j]} = {d.sh[j]}"]})
        diff = (sa - sb) % mod
        self._step(
            "cmp", f"Compare (1): d = {X.name} - {Y.name}  (free)",
            f"{X.name} < {Y.name} exactly when d = {X.name} - {Y.name} is negative. In the ring, a negative number has its TOP bit (bit {k - 1}) set to 1. "
            "So the whole comparison is: find the top bit of d. Computing d is free (it is a subtraction).",
            [f"Negative numbers are stored two's-complement: -1 = {'1' * k}, so the top bit is the sign.",
             f"Problem: d is spread over three pieces d₀ + d₁ + d₂. The top bit of a SUM depends on carries from all lower bits, so we cannot just look at the pieces."],
            parties, god=[f"True d = {self.sg(sa)} - {self.sg(sb)} = {self.sg(diff)}  (bits {self.bn(diff)}).  Top bit = {diff >> (k - 1)}, so {X.name} < {Y.name} is {'true' if diff >> (k - 1) else 'false'}."])

        # C2: boolean sharing of each additive piece (free)
        D = []
        for j in range(3):
            sh = [0, 0, 0]
            sh[j] = d.sh[j]
            D.append(Shared(sh, "B", f"D{j}", k))
        parties = []
        for i in range(3):
            parties.append({"holds": self._holds(i, *[(Dj, True) for Dj in D]),
                            "does": [f"I know d{sub(i)} and d{sub(i + 1)}, so I can write them as bit strings.",
                                     f"D{i} = d{sub(i)} is shared as ({self.bn(D[i].sh[i])}, 0, 0): only P{sub(i)} and P{sub(i - 1)} know it."]})
        self._step(
            "cmp", "Compare (2): treat each piece as a bit string (free)",
            "We now need bit-level operations on the pieces. Each piece d_j is already known to two parties, so it can be turned into a boolean sharing "
            "(XOR sharing) for free: put d_j in share j and zeros in the other two shares.",
            [f"Boolean sharing: value = share₀ ⊕ share₁ ⊕ share₂  (XOR instead of +).",
             f"Bits are written left to right from bit {k - 1} (top) down to bit 0.",
             "Goal: compute D0 + D1 + D2 (mod 2^k) on bits, and read its top bit."],
            parties, god=[f"D0 ⊕ ... wait: D{j} = d{sub(j)} = {self.bn(d.sh[j])}" for j in range(3)])
        # replace the awkward god line above with a clean one
        self.steps[-1]["god"] = [f"D{j} = d{sub(j)} = {self.bn(d.sh[j])}" for j in range(3)] + \
                                [f"D0 + D1 + D2 = {self.bn(sum(d.sh) % mod)} (mod 2^{k}) = d"]

        # C3: carry-save: 3 numbers -> 2 numbers (1 AND round)
        D0, D1, D2 = D
        S = Shared([a ^ b ^ c for a, b, c in zip(D0.sh, D1.sh, D2.sh)], "B", "S", k)
        U = self._xor(D0, D2, "U")
        V = self._xor(D1, D2, "V")
        (W,) = self._and_round(
            [(U, V, "W", m)], "cmp", "Compare (3): add three numbers into two (carry-save)",
            "Adding three numbers bit by bit: the sum bit is the XOR of the three bits (free), and the carry bit is 1 when at least two of the three bits are 1 (the 'majority'). "
            "Majority(a,b,c) = ((a⊕c) ∧ (b⊕c)) ⊕ c, which needs ONE secret AND. That AND is the first real communication of the comparison.",
            ["S = D0 ⊕ D1 ⊕ D2   (free: the sum bits)",
             "U = D0 ⊕ D2,   V = D1 ⊕ D2   (free)",
             "carry C = (U ∧ V) ⊕ D2, then shifted left by one bit: T = C ≪ 1   (AND = the multiplication protocol, bit by bit)",
             "Now D0 + D1 + D2 = S + T (mod 2^k): three numbers became two."], show=(S,))
        Mj = Shared([w ^ dd for w, dd in zip(W.sh, D2.sh)], "B", "C", k)
        T = self._shl(Mj, 1, "T")
        chk = (self._secret(S) + self._secret(T)) % mod
        self.steps[-1]["god"].append(f"S = {self.bn(self._secret(S))},  T = C≪1 = {self.bn(self._secret(T))},  S + T = {self.bn(chk)} = d {self._ok(chk == diff, 'carry-save identity')}")

        # C4: generate / propagate
        P = self._xor(S, T, "P")
        P0 = P
        (G,) = self._and_round(
            [(S, T, "G", m)], "cmp", "Compare (4): generate carries (G = S ∧ T)",
            "Now add the two numbers S and T. At each bit position, a carry is GENERATED when both bits are 1 (G = S ∧ T), and a carry PROPAGATES when exactly one of them is 1 (P = S ⊕ T, free). "
            "We only need one thing from this addition: the carry that arrives at the top bit.",
            ["G_i = S_i ∧ T_i   (1 AND round)", "P_i = S_i ⊕ T_i   (free)"])
        P = self._xor(S, T, "P")
        P0 = P
        self.steps[-1]["god"].append(f"P = S ⊕ T = {self.bn(self._secret(P))}")

        # C5: prefix levels
        dist, level = 1, 0
        while dist < m:
            level += 1
            Gs = self._shl(G, dist, f"G≪{dist}")
            items = [(P, Gs, "G'", m - dist)]
            do_p = 2 * dist < m
            if do_p:
                Ps = self._shl(P, dist, f"P≪{dist}")
                items.append((P, Ps, "P'", m - dist))
            res = self._and_round(
                items, "cmp", f"Compare (5.{level}): carry look-ahead, distance {dist}",
                f"A carry can travel through many bits (a long run of propagate positions). Instead of waiting for it bit by bit (k rounds), "
                f"we combine neighbours at distance {dist}: position i learns whether a carry coming from {dist} positions below reaches it. "
                f"Doubling the distance each round covers all {m} lower bits in about log₂({m}) rounds.",
                [f"G_i ← G_i ⊕ (P_i ∧ G_(i-{dist}))     ('either I generate, or I propagate what came from {dist} below')",
                 (f"P_i ← P_i ∧ P_(i-{dist})            ('I propagate only if both groups propagate')" if do_p
                  else "P is not needed after this level, so it is skipped."),
                 f"Both ANDs run in the same round. Positions below {dist} have nothing to combine, so only {m - dist} gates per AND."])
            Gn = Shared([g ^ a for g, a in zip(G.sh, res[0].sh)], "B", "G", k)
            Gn.name = "G"
            if do_p:
                Pn = res[1]
                Pn.name = "P"
                P = Pn
            G = Gn
            self.steps[-1]["god"].append(f"After this level: G = {self.bn(self._secret(G))}")
            dist *= 2

        # C6: sign bit
        carry = Shared([(s >> (m - 1)) & 1 for s in G.sh], "B", "g", 1)
        top = Shared([(s >> (k - 1)) & 1 for s in P0.sh], "B", "p", 1)
        sign = Shared([a ^ b for a, b in zip(top.sh, carry.sh)], "B", "lt", 1)
        parties = []
        for i in range(3):
            parties.append({"holds": self._holds(i, (carry, False), (top, False), (sign, True)),
                            "does": [f"g{sub(i)} = bit {m - 1} of G{sub(i)} = {carry.sh[i]},  p{sub(i)} = bit {k - 1} of P{sub(i)} = {top.sh[i]}",
                                     f"lt{sub(i)} = p{sub(i)} ⊕ g{sub(i)} = {sign.sh[i]}"]})
        expected = int(self.si(sa) < self.si(sb))
        c = self._ok(self._secret(sign) == expected, "comparison result")
        self._step(
            "cmp", f"Compare (6): read the sign bit  ({X.name} < {Y.name})",
            f"The top bit of the sum is (top bit of P) ⊕ (carry arriving at the top). Both are already shared bits, so XOR-ing them is free. "
            f"The result is a SECRET bit: 1 means {X.name} < {Y.name}. No party knows it yet.",
            [f"sign = P_{k - 1} ⊕ G_{m - 1}   (free)",
             "A secret bit like this can feed later computations (for example 'did the neuron spike?') without ever being revealed."],
            parties, god=[f"True result: {self.sg(sa)} < {self.sg(sb)} is {bool(expected)} → bit {expected}; shares XOR to {self._secret(sign)} {c}"])
        return sign


def cost_mul(k):
    return {"rounds": 1, "messages": 3, "bits": 3 * k}


def cost_compare(k):
    """Cost of the textbook comparison above, computed from the same loops without running it."""
    m = k - 1
    and_bits, rounds, dist = 2 * m, 2, 1
    while dist < m:
        rounds += 1
        and_bits += m - dist
        if 2 * dist < m:
            and_bits += m - dist
        dist *= 2
    return {"rounds": rounds, "messages": 3 * rounds, "and_bits": and_bits, "bits": 3 * and_bits}


def _check_range(bits, a, b, op):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    for name, v in (("a", a), ("b", b)):
        if not lo <= v <= hi:
            raise ValueError(f"{name} = {v} does not fit in {bits} bits (allowed {lo}..{hi})")
    if op == "cmp" and not lo <= a - b <= hi:
        raise ValueError(f"a - b = {a - b} does not fit in {bits} bits, so the sign bit would be wrong. Choose closer values.")


def run(op, a, b, bits=8, seed=0):
    """One full demonstration: share a and b, apply op, open the result. Returns steps + costs."""
    if op not in ("add", "mul", "cmp"):
        raise ValueError("op must be add, mul or cmp")
    a, b, bits = int(a), int(b), int(bits)
    _check_range(bits, a, b, op)
    sim = MPC(bits, seed)
    text = {"add": "a + b", "mul": "a · b", "cmp": "whether a < b"}[op]
    sim.setup(a, b, text)
    A = sim.share_input(0, a, "a")
    B = sim.share_input(1, b, "b", new_round=False)
    if op == "add":
        R = sim.add(A, B, "c")
        value = sim.reveal(R, "a + b")
        expected = (a + b) % sim.mod
        shown = sim.sg(value)
        exp_shown = sim.sg(expected)
    elif op == "mul":
        R = sim.mul(A, B, "c")
        value = sim.reveal(R, "a · b")
        expected = (a * b) % sim.mod
        shown, exp_shown = sim.sg(value), sim.sg(expected)
    else:
        R = sim.compare_lt(A, B)
        value = sim.reveal(R, "[a < b]")
        expected = int(a < b)
        shown, exp_shown = str(value), str(expected)
    exact = {"add": a + b, "mul": a * b, "cmp": expected}[op]
    wrapped = op != "cmp" and not -(sim.mod // 2) <= exact < sim.mod // 2
    c = sim.cost
    total = {"rounds": sum(v["rounds"] for v in c.values()), "messages": sum(v["messages"] for v in c.values()),
             "bits": sum(v["bits"] for v in c.values())}
    return {"op": op, "bits": bits, "a": a, "b": b, "result": shown, "expected": exp_shown,
            "ok": shown == exp_shown, "wrapped": wrapped, "exact": exact,
            "steps": sim.steps, "cost": c, "total": total, "and_bits": sim.and_bits,
            "formula": cost_mul(bits) if op == "mul" else cost_compare(bits) if op == "cmp" else {"rounds": 0, "messages": 0, "bits": 0}}


def secrecy_demo(bits, secret_a, secret_b, n=2000, seed=0):
    """
    What does ONE party see? Share each secret n times and histogram (x0 + x1 mod 2^k), the sum of the
    two pieces party 0 holds. For both secrets the histogram is flat: the view carries no information.
    """
    mod = 1 << bits
    rng = random.Random(seed)
    nb = 16
    width = max(1, mod // nb)
    out = {}
    for key, secret in (("a", secret_a), ("b", secret_b)):
        h = [0] * nb
        for _ in range(n):
            x0, x1 = rng.randrange(mod), rng.randrange(mod)
            h[min(nb - 1, ((x0 + x1) % mod) // width)] += 1
        out[key] = h
    return {"bits": bits, "n": n, "bins": nb, "bin_width": width, "a": secret_a, "b": secret_b,
            "hist_a": out["a"], "hist_b": out["b"], "expected_per_bin": n / nb}


def snn_preview(T, H, n_in, bits=64):
    """
    Plaintext SNN operations expressed as secret-sharing primitives, with the cost of each primitive taken
    from the formulas above. An ESTIMATE from textbook formulas (not MP-SPDZ measurements); Stage 3 replaces it.
    """
    cm, cc = cost_mul(bits), cost_compare(bits)
    k = bits

    def row(name, primitive, count, per_bits, rounds, note, training=False):
        return {"name": name, "primitive": primitive, "count": count, "bits": count * per_bits, "rounds": rounds,
                "note": note, "training": training}

    rows = [
        row("Leak and add: u = β·u + I", "add / public constant", T * H, 0, 0,
            "Free. β is a public number, so the multiplication needs no communication."),
        row("Synaptic current I = W·s", "dot product (reshare once)", T * H, cm["bits"], T,
            f"Each output is a sum of {n_in} products, summed locally and reshared once. The spikes s are secret, so we cannot skip the silent inputs."),
        row("Threshold test u ≥ θ", "comparison", T * H, cc["bits"], T * cc["rounds"],
            f"{cc['rounds']} rounds each, {H} neurons in parallel, {T} timesteps one after another."),
        row("Winner-take-all (which neuron spiked)", "comparisons (argmax)", T * (H - 1), cc["bits"], T * cc["rounds"] * 7,
            f"At least {H - 1} comparisons per timestep, arranged as a tree of about 7 levels. Lower bound: selection of the winner is extra."),
        row("Reset: u ← u·(1 - spike)", "multiplication", T * H, cm["bits"], T,
            "Spike is a secret bit, so this is a secret multiplication."),
        row("STDP update, secret winner", "multiplication", 2 * n_in * H * T, cm["bits"], 2 * T,
            f"We must not reveal the winner, so EVERY weight is updated and multiplied by a one-hot mask: 2 × {n_in} × {H} products per timestep. In plaintext only about 4 neurons update.", training=True),
        row("clamp(0, w_max) if kept", "comparisons", 2 * n_in * H * T, cc["bits"], 2 * T * cc["rounds"],
            "Two comparisons per weight per timestep. Almost certainly has to be redesigned (e.g. no clamp).", training=True),
    ]
    for r in rows:
        r["megabytes"] = r["bits"] / 8 / 1e6
    return {"bits": bits, "T": T, "H": H, "n_in": n_in, "cost_mul": cm, "cost_compare": cc, "rows": rows}

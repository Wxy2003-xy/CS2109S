const helper = (c) => {
    let res = Array(26).fill(c);
    let i = 0;
    for (i = 0; i < 26; i++) {
        res[i] = res[i] + String.fromCharCode(i + 97);
    }
    return res;
};
const gen_list = (lst, len) => {
    if (len == 0) {
        return lst;
    }
    const tmp = lst.flatMap(c => helper(c));
    return gen_list(tmp, len - 1);
};
console.log(gen_list([""], 5));

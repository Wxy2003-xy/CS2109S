const helper = (c: string): string[] => {
    let res: string[] = Array(26).fill(c);
    for (let i = 0; i < 26; i++) {
        res[i] = res[i] + String.fromCharCode(i + 97);
        console.log(res[i]); // Debugging: Shows intermediate results
    }
    return res;
};

const gen_list = (lst: string[], len: number): string[] => {
    if (len == 0) {
        return lst;
    }
    const tmp = lst.flatMap(c => helper(c));
    return gen_list(tmp, len - 1);
};

let lst: string[] = [""]; // Initialize with a non-empty array
let i: number = 0;
let ss: string[] = gen_list(lst, 5); // Generate the list

for (i = 0; i < 100 && i < ss.length; i++) {
    console.log(ss[i] + "\n");
}

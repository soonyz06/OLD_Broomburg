class WorkerRegistry:
    def __init__(self):
        self._workers = {}

    def register(self, names):
        def decorator(fn):
            for name in names:
                self._workers[name] = fn
            return fn
        return decorator

    def get(self, name):
        return self._workers.get(name)
    
    def parse_args(self, args): 
        result = {}
        for arg in args:
            parts = re.split(r'\s*[=:]\s*', arg, 1)
            key = parts[0]
            value = parts[1] if len(parts) > 1 else None
            if value=="None" :
                value = None
            result[key] = value
        return result

    def parse_command(self, cmd): #tokenisation, operatator handling, command resolution, parsing, execution
        cmd = cmd.lower()
        parts = re.split(r'(?=-\w+)', cmd)
        cmd = parts[0]
        cmd = cmd.replace("_", "-")
        tokens = [x.strip().upper() for x in cmd.split(" ") if x.strip()]
        if not tokens:
            return None, None, None

        if tokens[-1] in registry._workers.keys():
            func = tokens.pop()
            params = tokens 
    else:
        func = "DES"
        params = tokens

    settings = {}
    for p in parts[1:]:
        matches = re.findall(r'-(\w+)\s+([^-]+)', p.strip())
        if matches:
            subsets = matches[0][1:]
            for subset in subsets:
                args = re.findall(r'\S+\s*=\s*\S+|\S+', subset) #find all non white-space word(s) allowing for " = " 
                args = parse_args(args)
            settings[matches[0][0]] = args
    return params, func, settings

#params, func, settings = parse_command(cmd)

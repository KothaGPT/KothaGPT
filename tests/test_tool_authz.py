from services.security.authz import Authorizer, Permission


def test_allow():
    a = Authorizer([Permission("calc")])
    assert a.authorize("calc").allowed

def test_deny_no_perm():
    a = Authorizer([])
    assert not a.authorize("calc").allowed

def test_budget():
    a = Authorizer([Permission("calc", budget=1)])
    assert a.authorize("calc").allowed
    assert not a.authorize("calc").allowed

def test_approval_required():
    a = Authorizer([Permission("code", requires_approval=True)])
    dec = a.authorize("code")
    assert not dec.allowed and dec.requires_approval

def test_audit():
    a = Authorizer([Permission("calc")])
    a.authorize("calc")
    assert len(a.audit) == 1 and a.audit[0]["allowed"]
